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

// Direct-to-stderr trace, independent of the ggml log callback (tools like llama-bench install a
// callback that drops info logs). Enabled by GGML_NPU_VERBOSE=1 so a run can be *shown* to compute.
#define NPU_TRACE(...) do { if (getenv("GGML_NPU_VERBOSE")) { fprintf(stderr, "hpi-3720: " __VA_ARGS__); fflush(stderr); } } while (0)

struct ggml_backend_npu_context {
    hpi_device * hpi = nullptr;   // owned; opened at init, closed at free
    uint64_t     n_mul_mat = 0;   // Q8_0 mul_mat ops executed on this backend
    uint64_t     n_flop    = 0;   // 2*M*N*K summed, for the free-time summary
    bool         logged_first = false;
};

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

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
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

// --- backend interface -------------------------------------------------------------------------- //

static const char * ggml_backend_npu_get_name(ggml_backend_t backend) {
    return "hpi-3720";
    GGML_UNUSED(backend);
}

static void ggml_backend_npu_free(ggml_backend_t backend) {
    ggml_backend_npu_context * ctx = (ggml_backend_npu_context *) backend->context;
    if (ctx) {
        GGML_LOG_INFO("hpi-3720: backend ran %llu Q8_0 mul_mat op(s), %.3f GFLOP total\n",
                      (unsigned long long) ctx->n_mul_mat, (double) ctx->n_flop / 1e9);
        NPU_TRACE("backend ran %llu Q8_0 mul_mat op(s), %.3f GFLOP total\n",
                  (unsigned long long) ctx->n_mul_mat, (double) ctx->n_flop / 1e9);
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
    *free  = 0;
    *total = 0;
    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_npu_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
    GGML_UNUSED(dev);
}

// true only for the ops we compute: Q8_0 mul_mat and the free-of-charge structural ops.
static bool ggml_backend_npu_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
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
        /* .host_buffer           = */ false,
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

static ggml_backend_buffer_type_t ggml_backend_npu_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();   // host memory: the fake NPU reads/writes it directly
    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_npu_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_npu_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);
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
    /* .get_host_buffer_type = */ NULL,
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
