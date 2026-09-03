/*
 * ggml-npu.cpp — ggml backend for the Intel NPU 2.7 / VPU 3720.
 *
 * Structure follows ggml-blas: a minimal ACCEL device that uses host (CPU) memory and offloads a
 * single op — Q8_0 mul_mat — computing everything else on the CPU backend. The offloaded op is run
 * through the hpi-3720 Host-Platform Interface (hpi-3720/hpi.h). Today that resolves to the CPU
 * reference backend (the "fake" NPU), so results are correct; when the real VPU-3720 path is wired
 * in hpi_npu_3720.c, the same call site dispatches to silicon with no change here.
 *
 * Enough to make llama enumerate the NPU as a device (ggml_backend_dev_*), select it, and offload
 * Q8_0 weight matmuls to it (so -ngl / -ncmoe device placement has an NPU target).
 */
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-npu.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"   // ggml_backend_cpu_buffer_type / _from_ptr

#include "hpi-3720/hpi.h"

#include <cstring>
#include <cstdio>
#include <cstdlib>

static bool ggml_backend_npu_gpu_offload(void);   // NPU_OFFLOAD_GPU: whole-layer GPU offload vs default ACCEL

// Direct-to-stderr trace, independent of the ggml log callback (tools like llama-bench install a
// callback that drops info logs). Enabled by GGML_NPU_VERBOSE=1 so a run can be *shown* to compute.
#define NPU_TRACE(...) do { if (getenv("GGML_NPU_VERBOSE")) { fprintf(stderr, "hpi-3720: " __VA_ARGS__); fflush(stderr); } } while (0)

struct ggml_backend_npu_context {
    hpi_device * hpi = nullptr;   // owned; opened at init, closed at free
    ggml_backend_t cpu = nullptr; // internal CPU backend for GPU-passthrough: non-DPU nodes run here
    uint64_t     n_mul_mat = 0;   // Q8_0 mul_mat ops executed on the DPU path
    uint64_t     n_flop    = 0;   // 2*M*N*K summed, for the free-time summary
    uint64_t     n_m_gt1   = 0;   // ops with M>1 (prefill-shaped)
    uint64_t     n_cpu_pass = 0;  // nodes passed through to the internal CPU backend
    int64_t      max_m     = 0;   // largest M seen (diagnose prefill batching)
    bool         logged_first = false;
};

// A cacheable Q8_0 mul_mat shape: build_blob_cache authors an M=1 (decode) blob for any N%SLABCH==0,
// and an M<=256 (prefill) blob for K in {1024,2048}. Match that so GPU-passthrough routes exactly the
// ops with a DPU blob to the DPU, and sends the rest (lm_head N%256!=0, M>256, K=3072 prefill) to the
// fast CPU backend rather than the naive hpi CPU reference.
static bool ggml_backend_npu_dpu_cacheable(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT) return false;
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) return false;
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int64_t K = src0->ne[0], N = src0->ne[1], M = op->ne[1];
    if (N % 256 != 0 || K % 32 != 0) return false;
    if (M == 1) return true;                    // decode blob (any K)
    if (M <= 256 && (K == 1024 || K == 2048)) return true;   // prefill blob (fits CMX)
    return false;
}

// --- the offloaded op: Q8_0 mul_mat through the HPI --------------------------------------------- //

static void ggml_backend_npu_mul_mat(ggml_backend_npu_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // weights, Q8_0: [K=ne00, N=ne01, ne02, ne03]
    const struct ggml_tensor * src1 = dst->src[1]; // activations, F32: [K=ne10, M=ne11, ne12, ne13]

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ne00 == ne10);                       // K
    GGML_ASSERT(ne0  == ne01);                       // N
    GGML_ASSERT(ne1  == ne11);                        // M
    GGML_ASSERT(nb00 == (int64_t) ggml_type_size(GGML_TYPE_Q8_0)); // src0 block-contiguous
    GGML_ASSERT(nb10 == (int64_t) sizeof(float));                  // src1 contiguous rows
    GGML_ASSERT(nb0  == (int64_t) sizeof(float));                  // dst  contiguous rows

    const int64_t K = ne00, N = ne01, M = ne11;

    // ggml can emit empty mul_mat nodes (M=0 — e.g. a zero-token ubatch tail or an unused branch);
    // there is nothing to compute and the HPI layer rejects M<=0, so skip them exactly as the CPU
    // backend's 0-iteration loops do. (N/K are >0 for any real Q8_0 weight, guarded for safety.)
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    // grouped/broadcast batch dims, exactly as ggml mul_mat: src0's batch may be smaller than src1's
    const int64_t r2 = ne12 / ne02;
    const int64_t r3 = ne13 / ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13 / r3;
            const int64_t i02 = i12 / r2;

            const hpi_block_q8_0 * w = (const hpi_block_q8_0 *)((const char *) src0->data + i02*nb02 + i03*nb03);
            const float          * x = (const float *)         ((const char *) src1->data + i12*nb12 + i13*nb13);
            float                * y = (float *)               ((      char *) dst ->data + i12*nb2  + i13*nb3);

            const hpi_q8_0_gemm op = { M, N, K, w, x, y };
            const hpi_status st = hpi_q8_0_gemm_run(ctx->hpi, &op);
            if (st != HPI_OK) {
                GGML_LOG_ERROR("hpi-3720: Q8_0 gemm failed st=%d (%s) op='%s' M=%lld N=%lld K=%lld\n",
                               (int)st, hpi_status_str(st), dst->name, (long long)M, (long long)N, (long long)K);
            }
            GGML_ASSERT(st == HPI_OK);

            ctx->n_mul_mat++;
            ctx->n_flop += 2ull * (uint64_t)M * (uint64_t)N * (uint64_t)K;
            if (M > 1) ctx->n_m_gt1++;
            if (M > ctx->max_m) { ctx->max_m = M;
                NPU_TRACE("DPU mul_mat new max M=%lld (op '%s' N=%lld K=%lld)\n",
                          (long long)M, dst->name, (long long)N, (long long)K); }
            if (!ctx->logged_first) {
                ctx->logged_first = true;
                GGML_LOG_INFO("hpi-3720: computing Q8_0 mul_mat on the NPU backend "
                              "(first op '%s': M=%lld N=%lld K=%lld)\n",
                              dst->name, (long long)M, (long long)N, (long long)K);
                NPU_TRACE("computing Q8_0 mul_mat on the NPU backend (first op '%s': M=%lld N=%lld K=%lld)\n",
                          dst->name, (long long)M, (long long)N, (long long)K);
            }
        }
    }
}

static enum ggml_status ggml_backend_npu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_npu_context * ctx = (ggml_backend_npu_context *) backend->context;

    if (!ctx->cpu) {
        // ACCEL mode (default): the sched assigns us only our own ops (Q8_0 mul_mat + structural).
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    ggml_backend_npu_mul_mat(ctx, node);
                    break;
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_PERMUTE:
                case GGML_OP_TRANSPOSE:
                    break;
                default:
                    GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    // GPU-passthrough (NPU_OFFLOAD_GPU): -ngl placed whole layers here, so we get every op. Run the
    // cacheable Q8_0 mul_mats on the DPU and pass everything else (norms, rope, softmax, KV SET_ROWS,
    // lm_head, non-cacheable GEMMs) to the internal CPU backend — our buffers are host memory, so the
    // CPU computes them in place. Nodes are processed in order; the pending CPU batch is flushed before
    // each DPU op so data dependencies are honored.
    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_graph_overhead_custom(cgraph->size, false) + ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * gctx = ggml_init(ip);
    struct ggml_cgraph * batch = ggml_new_graph_custom(gctx, cgraph->size, false);

    // Independent consecutive DPU mul_mats (q/k/v share the norm input; gate/up share theirs) are
    // collected and submitted as ONE queue execute + sync (hpi_q8_0_gemm_batch), amortizing the per-op
    // submit+sync. A DPU op is batchable when it is a single hpi GEMM (ne12==ne13==1). Both the DPU and
    // CPU batches are flushed before any node that could read their outputs, preserving dependencies.
    const int DPU_BATCH = 32;
    hpi_q8_0_gemm        dpu_ops[DPU_BATCH];
    struct ggml_tensor * dpu_dst[DPU_BATCH];
    int n_dpu = 0;

    auto flush_cpu = [&]() {
        if (batch->n_nodes > 0) {
            ggml_backend_graph_compute(ctx->cpu, batch);
            ctx->n_cpu_pass += (uint64_t) batch->n_nodes;
            batch->n_nodes = 0;
        }
    };
    auto flush_dpu = [&]() {
        if (n_dpu > 0) {
            hpi_q8_0_gemm_batch(ctx->hpi, dpu_ops, n_dpu);
            for (int j = 0; j < n_dpu; j++) {
                const hpi_q8_0_gemm & o = dpu_ops[j];
                ctx->n_mul_mat++;
                ctx->n_flop += 2ull * (uint64_t) o.M * (uint64_t) o.N * (uint64_t) o.K;
                if (o.M > 1) ctx->n_m_gt1++;
                if (o.M > ctx->max_m) ctx->max_m = o.M;
            }
            n_dpu = 0;
        }
    };

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_NONE) continue;

        const bool batchable = ggml_backend_npu_dpu_cacheable(node) && node->ne[2] == 1 && node->ne[3] == 1;
        if (batchable) {
            flush_cpu();   // this DPU op may read a CPU-computed input -> ensure the CPU batch has run
            bool dep = false;   // never batch an op that reads a not-yet-executed batched DPU output
            for (int k = 0; k < n_dpu; k++) {
                if (node->src[0] == dpu_dst[k] || node->src[1] == dpu_dst[k]) { dep = true; break; }
            }
            if (dep || n_dpu == DPU_BATCH) flush_dpu();
            dpu_ops[n_dpu] = hpi_q8_0_gemm{
                node->ne[1], node->src[0]->ne[1], node->src[0]->ne[0],
                (const hpi_block_q8_0 *) node->src[0]->data,
                (const float *)          node->src[1]->data,
                (float *)                node->data,
            };
            dpu_dst[n_dpu] = node;
            n_dpu++;
        } else {
            flush_dpu();   // this node may read a batched DPU output -> execute the batch first
            if (ggml_backend_npu_dpu_cacheable(node)) {
                ggml_backend_npu_mul_mat(ctx, node);   // cacheable but ne12/ne13>1 (multi-GEMM) -> run individually
            } else {
                batch->nodes[batch->n_nodes++] = node;
            }
        }
    }
    flush_dpu();
    flush_cpu();
    ggml_free(gctx);
    return GGML_STATUS_SUCCESS;
}

// --- backend interface -------------------------------------------------------------------------- //

static const char * ggml_backend_npu_get_name(ggml_backend_t backend) {
    return "hpi-3720";
    GGML_UNUSED(backend);
}

static void ggml_backend_npu_free(ggml_backend_t backend) {
    ggml_backend_npu_context * ctx = (ggml_backend_npu_context *) backend->context;
    if (ctx) {
        GGML_LOG_INFO("hpi-3720: backend ran %llu Q8_0 mul_mat op(s) on the DPU (%llu with M>1, max M=%lld), "
                      "%llu node(s) passed to the CPU backend, %.3f GFLOP total\n",
                      (unsigned long long) ctx->n_mul_mat, (unsigned long long) ctx->n_m_gt1,
                      (long long) ctx->max_m, (unsigned long long) ctx->n_cpu_pass, (double) ctx->n_flop / 1e9);
        NPU_TRACE("backend ran %llu Q8_0 mul_mat op(s), %.3f GFLOP total\n",
                  (unsigned long long) ctx->n_mul_mat, (double) ctx->n_flop / 1e9);
        if (ctx->cpu) ggml_backend_free(ctx->cpu);
        hpi_close(ctx->hpi);
        delete ctx;
    }
    delete backend;
}

static struct ggml_backend_i npu_backend_i = {
    /* .get_name                = */ ggml_backend_npu_get_name,
    /* .free                    = */ ggml_backend_npu_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_npu_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_npu_guid(void) {
    // random, fixed for the life of the backend (ad1d = the NPU's PCI device id, as a mnemonic)
    static ggml_guid guid = { 0xad, 0x1d, 0x37, 0x20, 0x8b, 0x00, 0x86, 0x2e, 0x71, 0xf0, 0x0f, 0xa1, 0x33, 0x51, 0x2d, 0x27 };
    return &guid;
}

ggml_backend_t ggml_backend_npu_init(void) {
    ggml_backend_npu_context * ctx = new ggml_backend_npu_context;

    hpi_status st = HPI_OK;
    ctx->hpi = hpi_open(HPI_BACKEND_AUTO, &st);   // NPU_3720 if usable, else CPU reference
    if (!ctx->hpi) {
        GGML_LOG_ERROR("%s: hpi_open failed: %s\n", __func__, hpi_status_str(st));
        delete ctx;
        return NULL;
    }
    {
        hpi_device_info info;
        if (hpi_get_info(ctx->hpi, &info) == HPI_OK) {
            GGML_LOG_INFO("%s: NPU backend using hpi backend '%s' (hardware=%d)\n", __func__, info.name, info.is_hw);
        }
    }
    if (ggml_backend_npu_gpu_offload()) {
        // GPU-passthrough mode: an internal CPU backend computes every non-DPU node of the whole layers
        // that -ngl places on us (see ggml_backend_npu_graph_compute).
        ctx->cpu = ggml_backend_cpu_init();
        if (!ctx->cpu) {
            GGML_LOG_ERROR("%s: NPU_OFFLOAD_GPU set but ggml_backend_cpu_init() failed\n", __func__);
            hpi_close(ctx->hpi); delete ctx; return NULL;
        }
        GGML_LOG_INFO("%s: GPU-passthrough offload ON — whole layers via -ngl; cacheable Q8_0 mul_mats "
                      "on the DPU, all other ops on an internal CPU backend\n", __func__);
    }
    fprintf(stderr, "hpi-3720: offload mode = %s\n",
            ggml_backend_npu_gpu_offload() ? "GPU-passthrough (whole-layer via -ngl)" : "ACCEL (op-fallback)");

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_npu_guid(),
        /* .iface   = */ npu_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0),
        /* .context = */ ctx,
    };
    return backend;
}

bool ggml_backend_is_npu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_npu_guid());
}

// --- device interface --------------------------------------------------------------------------- //

// GPU-class whole-layer offload (NPU_OFFLOAD_GPU=1) vs the default ACCEL op-fallback. GPU-type makes
// -ngl place whole layers (full-M mul_mats) on the NPU — VERIFIED routing — but llama then also
// pre-allocates the layer's KV cache on our device buft and expects us to run its ops (SET_ROWS, ...),
// which we do not yet (that needs full layer-op coverage / CPU-passthrough). So it is OPT-IN and still
// WIP; default stays ACCEL (correct: opportunistic Q8_0 mul_mat offload, everything else on CPU).
static bool ggml_backend_npu_gpu_offload(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("NPU_OFFLOAD_GPU"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v == 1;
}

static const char * ggml_backend_npu_device_get_name(ggml_backend_dev_t dev) {
    // the name `--device <name>` matches (case-insensitive). Single device, so no index suffix
    // (cf. Hexagon: reg "HTP" -> devices "HTP0"/"HTP1"; here reg "NPU" -> device "hpi-3720").
    return "hpi-3720";
    GGML_UNUSED(dev);
}

static const char * ggml_backend_npu_device_get_description(ggml_backend_dev_t dev) {
    return "Intel NPU 2.7 / VPU 3720 (hpi-3720; CPU reference until silicon path lands)";
    GGML_UNUSED(dev);
}

static void ggml_backend_npu_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // Our "device memory" is host RAM (the buffer-type is host-backed; the DPU DMAs it). Report a
    // generous amount so -ngl layer-fitting places whole layers on us. NPU_MEM_MB overrides.
    const char * env = getenv("NPU_MEM_MB");
    size_t mb = env && env[0] ? (size_t) strtoull(env, NULL, 10) : (size_t) 96 * 1024;   // default 96 GiB
    *total = mb * 1024 * 1024;
    *free  = *total;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_npu_device_get_type(ggml_backend_dev_t dev) {
    // GPU (opt-in) puts whole layers on the NPU via -ngl (ggml-hexagon pattern); ACCEL (default) is
    // op-fallback only. See ggml_backend_npu_gpu_offload above.
    return ggml_backend_npu_gpu_offload() ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

// true only for the ops we compute: Q8_0 mul_mat and the free-of-charge structural ops.
static bool ggml_backend_npu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if (ggml_backend_npu_gpu_offload()) {
        // GPU-passthrough: cacheable Q8_0 mul_mats run on the DPU, every other op is delegated to the
        // internal CPU backend (ggml-cpu covers the full op set), so we accept any op — this is what
        // lets -ngl place whole layers (incl. their KV cache / SET_ROWS / norms) on the NPU device.
        return true;
    }
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_MUL_MAT: {
            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];
            return src0->type == GGML_TYPE_Q8_0 &&
                   src1->type == GGML_TYPE_F32  &&
                   op->type   == GGML_TYPE_F32  &&
                   ggml_is_contiguous(src0)     &&
                   ggml_is_contiguous(src1)     &&
                   (src0->ne[0] % 32 == 0);     // Q8_0 block-aligned K (always true, guarded anyway)
        }
        default:
            return false;
    }
    GGML_UNUSED(dev);
}

static bool ggml_backend_npu_device_offload_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    // worth offloading to the NPU exactly when we can compute it
    return ggml_backend_npu_device_supports_op(dev, op);
}

static void ggml_backend_npu_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_npu_device_get_name(dev);
    props->description = ggml_backend_npu_device_get_description(dev);
    props->type        = ggml_backend_npu_device_get_type(dev);
    ggml_backend_npu_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ ggml_backend_npu_gpu_offload(),   // GPU mode exposes a host buft (KV cache); ACCEL: none
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_npu_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_npu_init();
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

// A distinct, host-backed buffer-type OWNED by hpi-3720. Physically host memory (the DPU reads it by
// DMA), but a DISTINCT buft identity (not the CPU's) so ggml_backend_sched attributes weights placed
// here by -ngl to hpi-3720 and runs their mul_mats on the NPU — the ggml-hexagon whole-layer pattern,
// not the ACCEL op-fallback. No repack yet: weights stay Q8_0 in host memory, so build_blob still
// finds the per-tensor DPU blob by hashing op->w (a repack-on-set_tensor is the later optimization).
static const char * ggml_backend_npu_buft_get_name(ggml_backend_buffer_type_t buft) {
    return "hpi-3720";
    GGML_UNUSED(buft);
}
static ggml_backend_buffer_t ggml_backend_npu_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // delegate to the CPU host buffer (which owns + frees the memory), then re-label it as ours
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    if (buf) buf->buft = buft;
    return buf;
}
static size_t ggml_backend_npu_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    GGML_UNUSED(buft);
}
static bool ggml_backend_npu_buft_is_host(ggml_backend_buffer_type_t buft) {
    // FALSE on purpose: a device (non-host) buft is what makes ggml_backend_sched treat weights placed
    // here as DEVICE-resident and run their ops on hpi-3720 (a host-layout buft would be handled by the
    // CPU backend). The memory is still physically host RAM (delegated CPU buffer), so op->w stays
    // readable for build_blob and the DPU DMAs it — this is the integrated/unified-memory case.
    return false;
    GGML_UNUSED(buft);
}
static ggml_backend_buffer_type_t ggml_backend_npu_buffer_type(void) {
    static struct ggml_backend_buffer_type buft = {
        /* .iface = */ {
            /* .get_name       = */ ggml_backend_npu_buft_get_name,
            /* .alloc_buffer   = */ ggml_backend_npu_buft_alloc_buffer,
            /* .get_alignment  = */ ggml_backend_npu_buft_get_alignment,
            /* .get_max_size   = */ NULL,
            /* .get_alloc_size = */ NULL,
            /* .is_host        = */ ggml_backend_npu_buft_is_host,
        },
        /* .device  = */ NULL,
        /* .context = */ NULL,
    };
    buft.device = ggml_backend_reg_dev_get(ggml_backend_npu_reg(), 0);
    return &buft;
}

static ggml_backend_buffer_type_t ggml_backend_npu_device_get_buffer_type(ggml_backend_dev_t dev) {
    // GPU mode: our distinct device buft (-ngl places WEIGHTS here). ACCEL mode: plain host memory.
    return ggml_backend_npu_gpu_offload() ? ggml_backend_npu_buffer_type() : ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

// Host buffer-type for tensors that must stay CPU-runnable (the KV cache + its SET_ROWS/CPY updates,
// output, etc.). Weights go to our device buft (above) and run on the NPU; everything else in this
// host buft stays with the CPU backend — the ggml-hexagon two-buft split. Without this llama would
// pre-allocate the KV cache in our device buft and fail (we cannot run SET_ROWS).
static ggml_backend_buffer_type_t ggml_backend_npu_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_npu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_npu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // our own device buft (weights placed by -ngl) + any host buft (activations/inputs from the CPU).
    return buft == ggml_backend_npu_buffer_type() || ggml_backend_buft_is_host(buft);
    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_npu_device_i = {
    /* .get_name             = */ ggml_backend_npu_device_get_name,
    /* .get_description      = */ ggml_backend_npu_device_get_description,
    /* .get_memory           = */ ggml_backend_npu_device_get_memory,
    /* .get_type             = */ ggml_backend_npu_device_get_type,
    /* .get_props            = */ ggml_backend_npu_device_get_props,
    /* .init_backend         = */ ggml_backend_npu_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_npu_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_npu_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ ggml_backend_npu_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_npu_device_supports_op,
    /* .supports_buft        = */ ggml_backend_npu_device_supports_buft,
    /* .offload_op           = */ ggml_backend_npu_device_offload_op,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// --- backend reg interface ---------------------------------------------------------------------- //

static const char * ggml_backend_npu_reg_get_name(ggml_backend_reg_t reg) {
    return "NPU";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_npu_reg_get_device_count(ggml_backend_reg_t reg) {
    return 1;
    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_npu_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);
    static ggml_backend_device ggml_backend_npu_device = {
        /* .iface   = */ ggml_backend_npu_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };
    return &ggml_backend_npu_device;
    GGML_UNUSED(index);
}

static const struct ggml_backend_reg_i ggml_backend_npu_reg_i = {
    /* .get_name         = */ ggml_backend_npu_reg_get_name,
    /* .get_device_count = */ ggml_backend_npu_reg_get_device_count,
    /* .get_device       = */ ggml_backend_npu_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_npu_reg(void) {
    static struct ggml_backend_reg ggml_backend_npu_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_npu_reg_i,
        /* .context     = */ NULL,
    };
    return &ggml_backend_npu_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_npu_reg)
