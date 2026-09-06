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
#include "ggml-npu-deps.h"
#if defined(GGML_NPU_XE_LPG)
#include "ggml-npu-xe-lpg.h"
#endif

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
    uint64_t     n_split_tail = 0; // CPU row-split tails computed (GGML_NPU_SPLIT_PCT)
    bool         split_reported = false;
    int64_t      max_m     = 0;   // largest M seen (diagnose prefill batching)
    bool         logged_first = false;
    uint64_t     profile_graphs = 0;
#if defined(GGML_NPU_XE_LPG)
    ggml_backend_npu_xe_lpg * xe_lpg = nullptr;
#endif
};

// CMX FIT — MUST MIRROR re/build_blob_cache.py's cmx_need()/CMX_BUDGET EXACTLY.
//
// The M=1 template places two SLABCH weight slabs plus the fp16 output inside 1900 KiB of CMX. The
// authoring tool SKIPS any shape that does not fit. Until this predicate existed the backend did not
// know that, and the two silently disagreed — with a cost worse than the disagreement suggests:
// supports_op CLAIMED a shape with no blob, so instead of running on fast ggml-cpu the op fell through
// to the hpi SCALAR CPU reference (plain C11, no intrinsics). MEASURED on the unsloth target: 48
// CPU-ref fallbacks PER TOKEN, all M=1 N=2560 K=6144 (ssm_out, one per layer, 15.7M MACs each) — 48
// sizeable GEMMs per token taken off the fast path and run on the slowest path available.
//
// So this is not an optimization; refusing what we cannot author is strictly better than claiming it.
// K-split support must update both the author and these predicates before admitting larger K.
#define NPU_SLABCH     256                 // build_blob_cache.py SLABCH (its default, and our N%256 rule)
#define NPU_CMX_BUDGET (1900u * 1024u)     // build_blob_cache.py CMX_BUDGET
#define NPU_M1_K_MAX   8192                // measured unsliced limit; failure at 8208 is still unexplained

// Which cache generation NPU_BLOB_CACHE holds, detected once from its ready marker. v2 = one baked
// single-tile graph per weight (SLABCH 256 on one tile); v3 = shape-only two-tile programs with the
// weight image as a graph input: each tile streams N/2 channels in 128-channel slabs (256 when K<=1024,
// see re/build_blob_cache.py), so the CMX need is computed per tile. Only the geometry differs here;
// hpi_npu3720_blob.c still validates every entry on lookup and declines what the cache lacks.
static int ggml_backend_npu_cache_generation(void) {
    static int generation = 0;
    if (generation) return generation;
    generation = 2;
    const char *cache = getenv("NPU_BLOB_CACHE");
    if (cache && cache[0]) {
        char path[1024];
        const int n = snprintf(path, sizeof path, "%s/cache-v3.ready", cache);
        if (n > 0 && (size_t) n < sizeof path) {
            FILE *f = fopen(path, "rb");
            if (f) { generation = 3; fclose(f); }
        }
    }
    return generation;
}

static size_t ggml_backend_npu_cmx_need(int64_t K, int64_t N) {
    const bool two_tile = ggml_backend_npu_cache_generation() == 3;
    const size_t slabch = two_tile && K > 1024 ? 128u : (size_t) NPU_SLABCH;
    const size_t n_tile = two_tile ? (size_t) N / 2u : (size_t) N;
    const size_t wbytes = two_tile && K <= 1024 ? 2u : 1u;     // v3 authors K<=1024 as exact FP16 weights
    const size_t SB = slabch * 16u + slabch * (size_t) K * wbytes;
    const size_t out = ((size_t) K * 2u + 0xFFFu) & ~(size_t) 0xFFFu;
    const size_t s0 = (out + n_tile * 2u + 0xFFFu) & ~(size_t) 0xFFFu;
    const size_t s1 = (s0 + SB + 0xFFFu) & ~(size_t) 0xFFFu;
    return s1 + SB;
}

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
    if (K <= 0 || K > NPU_M1_K_MAX || N <= 0 || N > NPU_CMX_BUDGET / 2 || M <= 0) return false;
    if (N % NPU_SLABCH != 0 || K % 32 != 0) return false;
    if (ggml_backend_npu_cmx_need(K, N) > NPU_CMX_BUDGET) return false;  // authoring tool skips it -> so must we
    {   // Diagnostic placement knobs: keep ops outside [GGML_NPU_MIN_K, GGML_NPU_MAX_K] on the CPU (numerics A/B).
        static int64_t min_k = -1, max_k = -1;
        if (min_k < 0) {
            const char * lo = getenv("GGML_NPU_MIN_K"); const char * hi = getenv("GGML_NPU_MAX_K");
            min_k = (lo && lo[0]) ? atoll(lo) : 0; max_k = (hi && hi[0]) ? atoll(hi) : INT64_MAX;
        }
        if (K < min_k || K > max_k) return false;
    }
    if (M == 1) return true;                    // decode blob within measured K and CMX limits
    if (M <= 8 && ggml_backend_npu_cache_generation() == 3) return true;   // v3 8-row program (MTP verify batches)
    if (M <= 256 && (K == 1024 || K == 2048)) return true;   // prefill blob (fits CMX)
    return false;
}

// MoE expert matmul: true when a MUL_MAT_ID's experts are the Q8_0 shape our DPU path handles. The
// experts are a 3D stack [K, N, n_expert]; each routed (token, expert) is one M=1 Q8_0 gemm (below).
// Sub-8-bit experts (Q4_K/IQ4_NL/... — every cached MoE today) return false here and stay on the CPU
// passthrough, correct but unaccelerated, until the native-4-bit stream slab lands (Path B).
static bool ggml_backend_npu_mmid_cacheable(const struct ggml_tensor * op) {
    if (op->op != GGML_OP_MUL_MAT_ID) return false;
    const struct ggml_tensor * src0 = op->src[0];   // experts [K, N, n_expert]
    const struct ggml_tensor * src1 = op->src[1];   // activations [K, ne11, n_tokens]
    const struct ggml_tensor * ids  = op->src[2];   // expert ids [n_expert_used, n_tokens]
    if (!src0 || !src1 || !ids) return false;
    if (src0->type != GGML_TYPE_Q8_0 || src1->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) return false;
    if (ids->type != GGML_TYPE_I32) return false;
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) return false;
    const int64_t K = src0->ne[0], N = src0->ne[1];
    if (K <= 0 || K > NPU_M1_K_MAX || N <= 0 || N > NPU_CMX_BUDGET / 2) return false;
    if (N % NPU_SLABCH != 0 || K % 32 != 0) return false;
    if (ggml_backend_npu_cmx_need(K, N) > NPU_CMX_BUDGET) return false;  // same CMX limit as the 2D path

    // MEASURED 2026-09-04 (unsloth Qwen3.8-Flash-Next): claiming these is a NET LOSS today, so it is
    // opt-in. build_blob_cache.py authors blobs for 2D tensors ONLY (it skips len(ne)!=2), so a 3D
    // expert stack NEVER has a blob. Claiming the op therefore routes it to the hpi scalar CPU
    // reference, which is far slower than ggml-cpu -- we take work away from the fast path and do it
    // slowly. Confirmed on the box: every logged blob MISS in the first NPU perplexity attempt was an
    // expert slice of blk.2.ffn_down_exps (8/8 of the logged w0 values matched expert slices), M=1
    // N=2560 K=640, all falling back to the scalar reference.
    //
    // Note -ncmoe does NOT protect against this: supports_buft accepts any host buft, so experts kept
    // on the CPU by --n-cpu-moe are still schedulable to us. The gate has to be here.
    //
    // Set GGML_NPU_MMID=1 to re-enable once per-expert blobs exist (a real design question: 512
    // experts x N layers is far too many baked graphs -- this wants weights-as-input / A2, not a blob
    // per expert).
    {
        static int mmid = -1;
        if (mmid < 0) { const char * e = getenv("GGML_NPU_MMID"); mmid = (e && e[0] && e[0] != '0') ? 1 : 0; }
        if (!mmid) return false;
    }
    return true;                                     // per-expert rows are M=1 (decode blob, any K)
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
    GGML_ASSERT(nb00 == ggml_type_size(GGML_TYPE_Q8_0)); // src0 block-contiguous
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

// --- MoE experts: MUL_MAT_ID as one M=1 Q8_0 gemm per routed (token, expert) --------------------- //

static void ggml_backend_npu_mul_mat_id(ggml_backend_npu_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0]; // experts, Q8_0: [K=ne00, N=ne01, n_expert=ne02]
    const struct ggml_tensor * src1 = dst->src[1]; // activations, F32: [K=ne10, ne11, n_tokens=ne12]
    const struct ggml_tensor * ids  = dst->src[2]; // expert ids, I32: [n_expert_used=ne0, n_tokens=ne1]

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src0->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(ne00 == ne10);                       // K
    GGML_ASSERT(ne0  == ne01);                       // N (dst->ne[0])
    GGML_ASSERT(nb00 == ggml_type_size(GGML_TYPE_Q8_0)); // expert rows block-contiguous
    GGML_ASSERT(nb10 == sizeof(float));                  // src1 contiguous rows

    const int64_t K = ne00, N = ne01, n_expert = ne02;
    const int64_t n_ids = ids->ne[0];                // experts used per token
    const int64_t n_tok = ids->ne[1];                // tokens

    // One submit per token: its n_ids selected experts are INDEPENDENT M=1 gemms (distinct expert
    // matrices), so they batch into one queue execute+sync. Indexing mirrors ggml-cpu's reference
    // (ggml_compute_forward_mul_mat_id) exactly: expert e = ids[id, iid1]; its [K,N] matrix at
    // src0 + e*nb02 applies to src1 row (i11 = id % ne11, iid1) -> dst column (id, iid1).
    enum { NPU_MMID_MAX = 160 };                     // n_expert_used is small (<=16 in practice)
    hpi_q8_0_gemm ops[NPU_MMID_MAX];
    for (int64_t iid1 = 0; iid1 < n_tok; iid1++) {
        int nops = 0;
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t e = *(const int32_t *)((const char *) ids->data + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(e >= 0 && e < n_expert);
            const int64_t i11 = id % ne11;
            const hpi_block_q8_0 * w = (const hpi_block_q8_0 *)((const char *) src0->data + e*nb02);
            const float          * x = (const float *)         ((const char *) src1->data + i11*nb11 + iid1*nb12);
            float                * y = (float *)               ((      char *) dst ->data + id*nb1  + iid1*nb2);
            GGML_ASSERT(nops < NPU_MMID_MAX);
            ops[nops++] = hpi_q8_0_gemm{ 1, N, K, w, x, y };
        }
        if (nops == 0) continue;
        const hpi_status st = hpi_q8_0_gemm_batch(ctx->hpi, ops, nops);
        if (st != HPI_OK) {
            GGML_LOG_ERROR("hpi-3720: MUL_MAT_ID batch failed st=%d (%s) op='%s' experts=%d N=%lld K=%lld\n",
                           (int)st, hpi_status_str(st), dst->name, nops, (long long)N, (long long)K);
        }
        GGML_ASSERT(st == HPI_OK);
        ctx->n_mul_mat += (uint64_t) nops;
        ctx->n_flop    += 2ull * (uint64_t) nops * (uint64_t) N * (uint64_t) K;
    }
}

static enum ggml_status ggml_backend_npu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_npu_context * ctx = (ggml_backend_npu_context *) backend->context;
    hpi_profile before = {};
    GGML_ASSERT(hpi_get_profile(ctx->hpi, &before) == HPI_OK);
    const int64_t start_us = before.enabled ? ggml_time_us() : 0;
    double cpu_ms = 0.0;
    uint64_t cpu_batches = 0, overlap_candidates = 0;
    const char *overlap_env = getenv("GGML_NPU_OVERLAP");
    const bool overlap = overlap_env && overlap_env[0] && overlap_env[0] != '0';
    auto report_profile = [&]() {
        if (!before.enabled) return;
        hpi_profile after = {};
        GGML_ASSERT(hpi_get_profile(ctx->hpi, &after) == HPI_OK);
        const double wall_ms = (double)(ggml_time_us() - start_us) / 1000.0;
        const double prepare = after.prepare_ms - before.prepare_ms;
        const double input = after.input_ms - before.input_ms;
        const double submit = after.submit_ms - before.submit_ms;
        const double sync = after.sync_ms - before.sync_ms;
        const double output = after.output_ms - before.output_ms;
        const double fallback = after.fallback_ms - before.fallback_ms;
        // A scheduler graph view can cover part of a token. Normalize at the benchmark boundary.
        fprintf(stderr, "NPU_PROFILE graph=%llu nodes=%d wall_ms=%.6f cpu_ms=%.6f prepare_ms=%.6f input_ms=%.6f submit_ms=%.6f sync_ms=%.6f output_ms=%.6f fallback_ms=%.6f other_ms=%.6f cpu_batches=%llu submissions=%llu graphs=%llu fallback_ops=%llu cache_misses=%llu lookup_ms=%.6f build_ms=%.6f init_ms=%.6f record_ms=%.6f warm_hits=%llu negative_hits=%llu shape_builds=%llu command_records=%llu list_reuses=%llu host_work_ms=%.6f host_work_calls=%llu overlap_candidates=%llu\n",
                (unsigned long long)++ctx->profile_graphs, cgraph->n_nodes,
                wall_ms, cpu_ms, prepare, input, submit, sync, output, fallback,
                wall_ms - cpu_ms - prepare - input - submit - sync - output - fallback,
                (unsigned long long)cpu_batches,
                (unsigned long long)(after.submissions - before.submissions),
                (unsigned long long)(after.graphs - before.graphs),
                (unsigned long long)(after.fallback_ops - before.fallback_ops),
                (unsigned long long)(after.cache_misses - before.cache_misses),
                after.lookup_ms - before.lookup_ms, after.build_ms - before.build_ms,
                after.init_ms - before.init_ms, after.record_ms - before.record_ms,
                (unsigned long long)(after.warm_hits - before.warm_hits),
                (unsigned long long)(after.negative_hits - before.negative_hits),
                (unsigned long long)(after.shape_builds - before.shape_builds),
                (unsigned long long)(after.command_records - before.command_records),
                (unsigned long long)(after.list_reuses - before.list_reuses),
                after.host_work_ms - before.host_work_ms,
                (unsigned long long)(after.host_work_calls - before.host_work_calls),
                (unsigned long long)overlap_candidates);
    };

    if (!ctx->cpu) {
        // ACCEL mode (default): the sched assigns us only our own ops (Q8_0 mul_mat + structural).
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) continue;
            switch (node->op) {
                case GGML_OP_MUL_MAT:
                    ggml_backend_npu_mul_mat(ctx, node);
                    break;
                case GGML_OP_MUL_MAT_ID:
                    ggml_backend_npu_mul_mat_id(ctx, node);
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
        report_profile();
        return GGML_STATUS_SUCCESS;
    }

    // GPU-passthrough (NPU_OFFLOAD_GPU): -ngl placed whole layers here, so we get every op. Run the
    // cacheable Q8_0 mul_mats on the DPU and pass everything else (norms, rope, softmax, KV SET_ROWS,
    // lm_head, non-cacheable GEMMs) to the internal CPU backend — our buffers are host memory, so the
    // CPU computes them in place. Nodes are processed in order; the pending CPU batch is flushed before
    // each DPU op so data dependencies are honored.
    // CAPACITY MUST COME FROM n_nodes, NOT size. ggml_backend_sched hands each backend a GRAPH VIEW
    // (ggml-backend.cpp:1471 -> ggml_graph_view), and a view has `size = 0` by construction
    // (ggml.c:7446) because it cannot grow. Sizing this scratch graph by cgraph->size therefore
    // allocated a ZERO-CAPACITY node array, and `batch->nodes[batch->n_nodes++] = node` below wrote
    // straight past it into the ggml context buffer.
    //
    // That was the STATUS_HEAP_CORRUPTION (0xC0000374): the out-of-bounds writes land in adjacent heap,
    // so every flush "succeeds" and the process dies later — at ggml_free(gctx) freeing a smashed
    // block, or at some unrelated allocation. It is also what the earlier "segfault at node[7900]"
    // really was: 7900 node pointers written into a zero-capacity array.
    //
    // n_nodes is the exact bound: we append at most one entry per node of this graph, and flushes only
    // ever reset n_nodes to 0. Guard the append anyway — silently corrupting the heap is the worst
    // possible failure mode for a bound we believe is correct.
    const int batch_cap = cgraph->n_nodes > 0 ? cgraph->n_nodes : 1;
    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_graph_overhead_custom(batch_cap, false) + ggml_tensor_overhead()
                            + ggml_graph_overhead_custom(32, false) + 2u * 512u * ggml_tensor_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    struct ggml_context * gctx = ggml_init(ip);
    struct ggml_cgraph * batch = ggml_new_graph_custom(gctx, batch_cap, false);

    // Independent consecutive DPU mul_mats (q/k/v share the norm input; gate/up share theirs) are
    // collected and submitted as ONE queue execute + sync (hpi_q8_0_gemm_batch), amortizing the per-op
    // submit+sync. A DPU op is batchable when it is a single hpi GEMM (ne12==ne13==1). Both the DPU and
    // CPU batches are flushed before any node that could read their outputs, preserving dependencies.
    const int DPU_BATCH = 32;
    hpi_q8_0_gemm        dpu_ops[DPU_BATCH];
    struct ggml_tensor * dpu_dst[DPU_BATCH];
    int n_dpu = 0;

    // CPU/NPU ROW SPLIT (GGML_NPU_SPLIT_PCT=p, default off). At M=1 every DPU op is a weight stream and
    // the CPU idles in sync_ms; both engines are DDR-bound on the same DIMMs at ~30-36 GB/s each while
    // the DIMMs give ~70. So for a large decode GEMV the NPU takes rows [0, n_npu) -- a v3 cache entry
    // authored for exactly those rows (its digest is the digest of those bytes, so the provider needs
    // no change) -- and the CPU computes rows [n_npu, N) into the same dst inside the overlap callback
    // while the NPU streams. n_npu = floor(N*p/100 / 512) * 512 must match re/build_blob_cache.py.
    static int split_pct = -1;
    if (split_pct < 0) {
        const char * e = getenv("GGML_NPU_SPLIT_PCT");
        split_pct = (e && e[0]) ? atoi(e) : 0;
        if (split_pct < 0 || split_pct > 99) split_pct = 0;
    }
    struct npu_tail { struct ggml_tensor * src0; struct ggml_tensor * src1; struct ggml_tensor * dst; int64_t n_npu; };
    npu_tail tails[DPU_BATCH];
    int n_tail = 0;
    int split_budget = 512;                          // tensors we may still create in gctx this call
    struct ggml_cgraph * tail_graph = split_pct ? ggml_new_graph_custom(gctx, 32, false) : NULL;
    auto run_tails = [&]() {
        if (n_tail == 0) return;
        tail_graph->n_nodes = 0;
        for (int t = 0; t < n_tail; ++t) {
            struct ggml_tensor * s0 = tails[t].src0;
            struct ggml_tensor * s1 = tails[t].src1;
            struct ggml_tensor * d = tails[t].dst;
            const int64_t K = s0->ne[0], N = s0->ne[1], n0 = tails[t].n_npu;
            struct ggml_tensor * w = ggml_view_2d(gctx, s0, K, N - n0, s0->nb[1], (size_t) n0 * s0->nb[1]);
            w->buffer = s0->buffer;
            struct ggml_tensor * o = ggml_mul_mat(gctx, w, s1);
            o->data = (char *) d->data + (size_t) n0 * sizeof(float);   // M == 1: the tail rows are contiguous
            o->buffer = d->buffer;
            o->flags |= GGML_TENSOR_FLAG_COMPUTE;              // ggml-cpu skips nodes without it (ggml-cpu.c graph_compute_thread)
            tail_graph->nodes[tail_graph->n_nodes++] = o;
        }
        const int64_t t0 = before.enabled ? ggml_time_us() : 0;
        GGML_ASSERT(ggml_backend_graph_compute(ctx->cpu, tail_graph) == GGML_STATUS_SUCCESS);
        if (before.enabled) cpu_ms += (double)(ggml_time_us() - t0) / 1000.0;
        ctx->n_split_tail += (uint64_t) n_tail;
        n_tail = 0;
    };

    auto flush_cpu = [&]() {
        if (batch->n_nodes > 0) {
            if (getenv("GGML_NPU_TRACE_NODES")) {
                fprintf(stderr, "hpi-3720:   >> flush_cpu %d node(s)\n", batch->n_nodes); fflush(stderr);
            }
            const int64_t cpu_start = before.enabled ? ggml_time_us() : 0;
            GGML_ASSERT(ggml_backend_graph_compute(ctx->cpu, batch) == GGML_STATUS_SUCCESS);
            if (before.enabled) {
                cpu_ms += (double)(ggml_time_us() - cpu_start) / 1000.0;
                cpu_batches++;
            }
            if (getenv("GGML_NPU_TRACE_NODES")) { fprintf(stderr, "hpi-3720:   << flush_cpu ok\n"); fflush(stderr); }
            ctx->n_cpu_pass += (uint64_t) batch->n_nodes;
            batch->n_nodes = 0;
        }
    };
    auto flush_dpu = [&]() {
        if (n_dpu > 0) {
            if (getenv("GGML_NPU_TRACE_NODES")) {
                fprintf(stderr, "hpi-3720:   >> flush_dpu %d op(s), first M=%lld N=%lld K=%lld\n",
                        n_dpu, (long long) dpu_ops[0].M, (long long) dpu_ops[0].N, (long long) dpu_ops[0].K);
                fflush(stderr);
            }
            hpi_status results[DPU_BATCH];
            if (n_tail > 0) {
                auto work = [](void *user) { (*static_cast<decltype(run_tails) *>(user))(); };
                GGML_ASSERT(hpi_q8_0_gemm_batch_overlap(ctx->hpi, dpu_ops, n_dpu, results,
                            work, &run_tails) == HPI_OK);
            } else if (overlap && batch->n_nodes > 0) {
                auto work = [](void *user) { (*static_cast<decltype(flush_cpu) *>(user))(); };
                GGML_ASSERT(hpi_q8_0_gemm_batch_overlap(ctx->hpi, dpu_ops, n_dpu, results,
                            work, &flush_cpu) == HPI_OK);
            } else {
                GGML_ASSERT(hpi_q8_0_gemm_batch_try(ctx->hpi, dpu_ops, n_dpu, results) == HPI_OK);
            }
            if (getenv("GGML_NPU_TRACE_NODES")) { fprintf(stderr, "hpi-3720:   << flush_dpu ok\n"); fflush(stderr); }
            for (int j = 0; j < n_dpu; j++) {
                if (results[j] == HPI_UNAVAILABLE) {
                    if (batch->n_nodes >= batch_cap) flush_cpu();
                    batch->nodes[batch->n_nodes++] = dpu_dst[j];
                    continue;
                }
                GGML_ASSERT(results[j] == HPI_OK);
                const hpi_q8_0_gemm & o = dpu_ops[j];
                ctx->n_mul_mat++;
                ctx->n_flop += 2ull * (uint64_t) o.M * (uint64_t) o.N * (uint64_t) o.K;
                if (o.M > 1) ctx->n_m_gt1++;
                if (o.M > ctx->max_m) ctx->max_m = o.M;
            }
            n_dpu = 0;
        }
    };

    // GGML_NPU_TRACE_NODES=1 prints every node BEFORE it is classified or executed, flushed, so a crash
    // inside the delegated CPU batch names the last node reached. GPU-passthrough hands us ops this path
    // has never seen on models beyond the plain transformer it was developed against (Flash-Next brings
    // SSM_SCAN/SSM_CONV, hyper-connections, the PLE gather), and a segfault with no output is otherwise
    // extremely expensive to localise.
    static int trace_nodes = -1;
    if (trace_nodes < 0) {
        const char * e = getenv("GGML_NPU_TRACE_NODES");
        trace_nodes = (e && e[0] && e[0] != '0') ? 1 : 0;
    }

#if defined(GGML_NPU_XE_LPG)
    bool xe_lpg_bypass = false;
#endif
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        if (trace_nodes) {
            fprintf(stderr, "hpi-3720: node[%d/%d] %-16s %-28s ne=[%lld,%lld,%lld,%lld]\n",
                    i, cgraph->n_nodes, ggml_op_desc(node), node->name,
                    (long long) node->ne[0], (long long) node->ne[1],
                    (long long) node->ne[2], (long long) node->ne[3]);
            fflush(stderr);
        }
        if (node->op == GGML_OP_NONE) continue;

#if defined(GGML_NPU_XE_LPG)
        const int xe_lpg_role = ggml_backend_npu_xe_lpg_role(ctx->xe_lpg, node);
        if (xe_lpg_bypass) {
            if (xe_lpg_role == 3) xe_lpg_bypass = false;
        } else if (xe_lpg_role) {
            flush_dpu();
            if (xe_lpg_role == 1) flush_cpu();
            if (xe_lpg_role == 1) {
                const int replace = ggml_backend_npu_xe_lpg_begin_replace(ctx->xe_lpg, cgraph, i);
                if (replace > 0) {
                    struct ggml_tensor * down = cgraph->nodes[i + 3];
                    if (ggml_backend_npu_xe_lpg_replace(ctx->xe_lpg, down)) {
                        i += 3;
                        continue;
                    }
                    GGML_ASSERT(batch->n_nodes == 0);
                    GGML_ASSERT(batch_cap >= 4);
                    for (int held = 0; held < 4; ++held) {
                        if (batch->n_nodes >= batch_cap) flush_cpu();
                        batch->nodes[batch->n_nodes++] = cgraph->nodes[i + held];
                    }
                    flush_cpu();
                    ggml_backend_npu_xe_lpg_abort(ctx->xe_lpg);
                    i += 3;
                    continue;
                }
                if (replace < 0) {
                    ggml_backend_npu_xe_lpg_abort(ctx->xe_lpg);
                    xe_lpg_bypass = true;
                }
            }
            if (xe_lpg_bypass) {
                // The current CPU node falls through to the normal classifier.
            } else {
                ggml_backend_npu_xe_lpg_capture(ctx->xe_lpg, node, xe_lpg_role);
                if (batch->n_nodes >= batch_cap) flush_cpu();
                batch->nodes[batch->n_nodes++] = node;
                if (xe_lpg_role == 3) {
                    flush_cpu();
                    ggml_backend_npu_xe_lpg_complete(ctx->xe_lpg, node);
                }
                continue;
            }
        }
#endif

        const bool batchable = ggml_backend_npu_dpu_cacheable(node) && node->ne[2] == 1 && node->ne[3] == 1;
        if (trace_nodes && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            const struct ggml_tensor * s0 = node->src[0];
            const struct ggml_tensor * s1 = node->src[1];
            fprintf(stderr, "hpi-3720:   ^ batchable=%d s0{%s data=%p buf=%s} s1{%s data=%p buf=%s} dst data=%p\n",
                    (int) batchable,
                    s0 ? ggml_type_name(s0->type) : "null", s0 ? s0->data : NULL,
                    (s0 && s0->buffer) ? ggml_backend_buffer_name(s0->buffer) : "NOBUF",
                    s1 ? ggml_type_name(s1->type) : "null", s1 ? s1->data : NULL,
                    (s1 && s1->buffer) ? ggml_backend_buffer_name(s1->buffer) : "NOBUF",
                    node->data);
            fflush(stderr);
        }
        if (batchable) {
            if (n_dpu && batch->n_nodes) flush_dpu();
            flush_cpu();   // this DPU op may read a CPU-computed input -> ensure the CPU batch has run
            bool dep = false;   // never batch an op that reads a not-yet-executed batched DPU output
            for (int k = 0; k < n_dpu; k++) {
                if (!ggml_npu_independent(node, dpu_dst[k])) { dep = true; break; }
            }
            if (dep || n_dpu == DPU_BATCH) {
                flush_dpu();
                flush_cpu(); // A dependency can have fallen back to the CPU batch.
            }
            dpu_ops[n_dpu] = hpi_q8_0_gemm{
                node->ne[1], node->src[0]->ne[1], node->src[0]->ne[0],
                (const hpi_block_q8_0 *) node->src[0]->data,
                (const float *)          node->src[1]->data,
                (float *)                node->data,
            };
            {
                const int64_t N = node->src[0]->ne[1], K = node->src[0]->ne[0], M = node->ne[1];
                if (split_pct && M == 1 && K > 1024 && N >= 4096 && split_budget >= 2) {
                    const int64_t n_npu = ((N * split_pct / 100) / 512) * 512;
                    if (n_npu >= 512 && n_npu < N) {
                        dpu_ops[n_dpu].N = n_npu;
                        tails[n_tail++] = npu_tail{ node->src[0], node->src[1], node, n_npu };
                        split_budget -= 2;
                    }
                }
            }
            dpu_dst[n_dpu] = node;
            n_dpu++;
        } else {
            bool independent = overlap && n_dpu > 0;
            if (independent && (ggml_backend_npu_dpu_cacheable(node) || ggml_backend_npu_mmid_cacheable(node)))
                independent = false;
            for (int k = 0; independent && k < n_dpu; ++k)
                independent = ggml_npu_independent(node, dpu_dst[k]);
            if (independent && ggml_npu_metadata_op(node->op)) continue;
            if (independent && batch->n_nodes < batch_cap) {
                batch->nodes[batch->n_nodes++] = node;
                if (!ggml_npu_metadata_op(node->op)) overlap_candidates++;
                continue;
            }
            flush_dpu();   // resolve device outputs before a dependent or side-effecting CPU op
            if (ggml_backend_npu_dpu_cacheable(node)) {
                flush_cpu();
                ggml_backend_npu_mul_mat(ctx, node);   // cacheable but ne12/ne13>1 (multi-GEMM) -> run individually
            } else if (ggml_backend_npu_mmid_cacheable(node)) {
                flush_cpu();                           // experts read CPU-computed activations -> flush first
                ggml_backend_npu_mul_mat_id(ctx, node);   // MoE experts on the DPU (one batched submit per token)
            } else {
                if (batch->n_nodes >= batch_cap) {        // must never happen (cap == n_nodes); never corrupt if it does
                    flush_cpu();
                }
                batch->nodes[batch->n_nodes++] = node;   // everything else (incl. sub-8-bit experts) -> CPU passthrough
            }
        }
    }
    flush_dpu();
    flush_cpu();
    ggml_free(gctx);
    report_profile();
    if (split_pct && !ctx->split_reported && ctx->n_split_tail) {
        ctx->split_reported = true;
        fprintf(stderr, "hpi-3720: CPU/NPU row split active (GGML_NPU_SPLIT_PCT=%d): %llu CPU tails so far\n",
                split_pct, (unsigned long long) ctx->n_split_tail);
    }
#if defined(GGML_NPU_XE_LPG)
    ggml_backend_npu_xe_lpg_report_graph(ctx->xe_lpg);
#endif
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
#if defined(GGML_NPU_XE_LPG)
        ggml_backend_npu_xe_lpg_destroy(ctx->xe_lpg);
#endif
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
        // Four threads beat larger internal pools in the Flash-Next decode sweep. The cause remains unmeasured.
        if (ctx->cpu) {
            int nth = 4;
            if (const char * e = getenv("GGML_NPU_CPU_THREADS")) {
                const int v = atoi(e);
                if (v > 0) nth = v;
            }
            ggml_backend_cpu_set_n_threads(ctx->cpu, nth);
            NPU_TRACE("internal CPU backend: %d threads\n", nth);
        }
        if (!ctx->cpu) {
            GGML_LOG_ERROR("%s: NPU_OFFLOAD_GPU set but ggml_backend_cpu_init() failed\n", __func__);
            hpi_close(ctx->hpi); delete ctx; return NULL;
        }
#if defined(GGML_NPU_XE_LPG)
        ctx->xe_lpg = ggml_backend_npu_xe_lpg_create(ctx);
#endif
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

// GPU-class whole-layer offload (NPU_OFFLOAD_GPU=1) vs the ACCEL op-fallback. GPU-type makes -ngl
// place whole layers (full-M mul_mats) on the NPU — VERIFIED routing. The old blocker here (llama
// pre-allocating the layer's KV cache on our device buft and expecting us to run SET_ROWS et al) is
// SOLVED: get_host_buffer_type below hands the KV cache back to the CPU, and graph_compute delegates
// every non-DPU node to the internal CPU backend — the ggml-hexagon two-buft split.
//
// GPU-passthrough is THE DIRECTION, not an experiment: every silicon measurement in re/FINDINGS.md
// (incl. the batched/unbatched A/B at :1635) was taken with NPU_OFFLOAD_GPU=1. The env var default
// stays 0 only so an unset run cannot surprise anyone with the no-mmap cost below — do NOT read that
// default as ACCEL being the plan.
//
// COST, and it is sharp: the device buft is is_host=false ON PURPOSE (that is what makes the sched
// route ops to us), so llama does NOT mmap weights placed here — it commits them to RAM. On a large
// model -ngl 999 without -ncmoe will try to commit the whole file. See scripts/run.sh.
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
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_npu_mmid_cacheable(op);   // Q8_0 MoE experts (sub-8-bit -> not us, stays on CPU)
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

// GGML_NPU_DISABLE=1 hides the device at REGISTRATION time (device count 0), which is the only way to
// keep this backend out of a run entirely.
//
// Why a registration-time opt-out is necessary: `--device none` does NOT exclude us. llama.cpp
// unconditionally initializes every ACCEL-type device and adds it to the scheduler
// (llama-context.cpp:341 "add ACCEL backends (such as BLAS)"), independently of the --device list --
// and in ACCEL mode our supports_op claims Q8_0 mul_mats, so we WOULD silently compute part of a run
// the user asked to be CPU-only. That is exactly how a "CPU baseline" gets quietly contaminated by the
// very DPU path it is meant to validate. Reporting no device is the honest off switch.
//
// (In GPU-passthrough mode we report DEVICE_TYPE_GPU, so --device does gate us; this env var covers
// both modes uniformly, so a baseline script does not have to know which mode it is in.)
static bool ggml_backend_npu_disabled(void) {
    static int v = -1;
    if (v < 0) { const char * e = getenv("GGML_NPU_DISABLE"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v == 1;
}

static size_t ggml_backend_npu_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_npu_disabled() ? 0 : 1;
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
