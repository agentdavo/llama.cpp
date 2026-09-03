/*
 * hpi_npu_3720.c — the real Intel NPU 2.7 / VPU 3720 backend for the Q8_0 GEMM offload.
 *
 * STATUS: PLUMBING WIRED, COMPUTE SEAM OPEN.
 *
 * When HPI_HAVE_NPU_3720 is defined (a Windows x64 build with the NPU present and the src/
 * direct-access toolkit — src/npu.h + the Level Zero / NPU-extension headers — on the include path),
 * this TU drives the device through the *proven* src/ ladder (npu_ze_load -> npu_dev_open ->
 * npu_graph_create -> AppendGraphInitialize/Execute -> readback; verified in src/ze_npu_graph.c on
 * real silicon). Everything the proven API supports is written out here: device lifecycle, a
 * per-shape graph/buffer cache, activation f32<->fp16 conversion, argument binding, submit, and
 * readback.
 *
 * The ONE thing that is not written is the thing that cannot be responsibly guessed from a Linux
 * sandbox with no NPU and no OpenVINO: step 2, "produce a native compiled blob that computes this
 * exact Q8_0 GEMM shape." That is the single seam `hpi_npu3720_build_blob()`. It is gated behind
 * HPI_NPU3720_BLOB_READY and returns HPI_UNAVAILABLE until implemented on the hardware box. Because
 * `npu_available()` is gated by the *same* macro, until the blob path is real this backend reports
 * unavailable, so HPI_BACKEND_AUTO always falls back to the CPU reference (which the ggml layer
 * asserts on: every hpi_q8_0_gemm_run must return HPI_OK, so we must never advertise a device we
 * cannot actually compute on). No guess is ever presented as a result — outer CLAUDE.md rule 5.
 *
 * !! IMPORTANT: the HPI_HAVE_NPU_3720 branch below has NOT been compiled anywhere yet — it needs the
 *    Windows Level Zero stack and src/npu.h, neither of which exists in the CI/sandbox that builds
 *    the default (CPU-reference) backend. Treat it as a reviewed-but-unbuilt scaffold: compile it on
 *    the box (cmake -DGGML_NPU_HW=ON, with NPU_SRC_DIR / LEVEL_ZERO_INCLUDE set), fix what the
 *    compiler finds, then implement the seam and flip HPI_NPU3720_BLOB_READY.
 */
#include "hpi_backend.h"

#if defined(HPI_HAVE_NPU_3720)

/* ---------------------------------------------------------------------------------------------- *
 *  Bring in the proven direct-access toolkit. This TU owns the npu.h implementation (exactly one
 *  translation unit may, per src/npu.h). NPU_NO_GDN drops the AVX2 GDN host math we don't use here;
 *  the full Level Zero layer (sections 3-5) is required, so NPU_NO_ZE must NOT be set.
 * ---------------------------------------------------------------------------------------------- */
#define NPU_IMPLEMENTATION
#define NPU_NO_GDN
#include "npu.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* How many distinct (weights, M, N, K) GEMM shapes we keep compiled+resident at once. qwen35 decode
 * touches on the order of a dozen distinct weight tensors x {prefill M, decode M=1}; the src/ loader
 * table exposes no zeCommandList/QueueDestroy and no graph pfnDestroy wrapper, so we deliberately do
 * NOT churn these per call — we build once per shape and keep them. Raise if a model needs more. */
#ifndef NPU3720_CACHE_MAX
#define NPU3720_CACHE_MAX 64
#endif

/* ---------------------------------------------------------------------------------------------- *
 *  The compute seam (step 2). NOT IMPLEMENTED.
 *
 *  Build (or fetch from a cache/OpenVINO compile) a native VPU-3720 blob computing, for this exact
 *  shape, Y[MxN] = X[MxK] . dequant(W)^T, and describe how to bind its runtime arguments.
 *
 *  Two viable near-term strategies (see re/FINDINGS.md 2026-09-02 "PER-CHANNEL DEQUANT COMPILES" and
 *  the outer CLAUDE.md "long game"); the blob author picks one and reports it via *io:
 *    (A) weights baked into the blob as folded fp16 constants (io->arg_w < 0). Simplest first bring-
 *        up: only X is a runtime input. Cost: no runtime dequant, full fp16 weights resident per
 *        shape (the constant-folding note). The cache key includes the W pointer for this reason.
 *    (B) weights as a runtime input with a SHAVE/compiler dequant kernel (io->arg_w >= 0). Note the
 *        hard constraint: the 3720 compiler only wires a PER-CHANNEL DequantizeOp; Q8_0's per-32-
 *        element block scale does not survive it, so this path needs host/SHAVE-side dequant or a
 *        per-channel re-scale. The plumbing below does not yet stage arbitrary weight encodings, so
 *        it currently rejects arg_w >= 0 (returns HPI_EDEVICE) — extend staging when (B) is built.
 *
 *  On success: *blob = malloc'd native blob (caller frees), *blob_len its size, *io filled.
 *  Until implemented: returns HPI_UNAVAILABLE.
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
    int arg_x;   /* graph argument index that receives activations X (>= 0) */
    int arg_y;   /* graph argument index that produces output Y (>= 0)      */
    int arg_w;   /* weights input index, or < 0 if weights are baked into the blob */
} hpi_npu3720_blob_io;

#if defined(HPI_NPU3720_BLOB_READY)
/* Provided by the hardware-side implementation (a separate TU compiled on the box). */
extern hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                         uint8_t **blob, size_t *blob_len,
                                         hpi_npu3720_blob_io *io);
#else
static hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                         uint8_t **blob, size_t *blob_len,
                                         hpi_npu3720_blob_io *io) {
    (void)op; (void)blob; (void)blob_len; (void)io;
    return HPI_UNAVAILABLE;   /* the one open piece — see the block comment above */
}
#endif

/* ---------------------------------------------------------------------------------------------- *
 *  Per-shape resident state: one compiled blob + its graph + its staged, device-visible buffers.
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
    const void *w_key;               /* W pointer this blob was built for (strategy A cache key) */
    int64_t     M, N, K;
    uint8_t    *blob;                /* owned */
    size_t      blob_len;
    npu_graph   g;
    hpi_npu3720_blob_io io;
    void       *x_mem;               /* device-visible input  buffer (bound to g.arg[io.arg_x]) */
    void       *y_mem;               /* device-visible output buffer (bound to g.arg[io.arg_y]) */
    int         initialized;         /* AppendGraphInitialize has run for this graph */
    int         in_use;
} npu3720_shape;

typedef struct {
    npu_ze                    z;
    npu_dev                   d;
    ze_command_queue_handle_t q;
    int                       q_ok;
    npu3720_shape             cache[NPU3720_CACHE_MAX];
    int                       ncache;
} npu3720_priv;

/* ---------------------------------------------------------------------------------------------- *
 *  available() / open() / close()
 * ---------------------------------------------------------------------------------------------- */

/* Cheap capability report. MUST be 0 whenever a gemm cannot produce a correct result, because
 * HPI_BACKEND_AUTO selects this backend on a true here and the ggml layer asserts every gemm is
 * HPI_OK. So it is gated on the compute seam being real; only then does it matter whether a VPU is
 * physically present (probed at open). */
static int npu_available(void) {
#if defined(HPI_NPU3720_BLOB_READY)
    return 1;
#else
    return 0;
#endif
}

static hpi_status npu_open(hpi_device *dev) {
    if (!dev) return HPI_EINVAL;
    npu3720_priv *p = (npu3720_priv *)calloc(1, sizeof *p);
    if (!p) return HPI_ENOMEM;

    /* Step 1: load the Level Zero loader and open the first VPU device (+ context + graph table). */
    if (npu_ze_load(&p->z, 0) != 0) { free(p); return HPI_EDEVICE; }
    if (npu_dev_open(&p->z, &p->d) != 0) { free(p); return HPI_EDEVICE; }
    if (!p->d.graph) {                      /* no ZE_GRAPH_EXT table -> cannot run compiled blobs */
        free(p);
        return HPI_EDEVICE;
    }

    /* One synchronous, non-turbo queue for the device's lifetime. Non-turbo on purpose: the turbo
     * clock + a metric streamer once hard-froze the machine (src/npu.h HAZARD); leave turbo to an
     * explicit, coordinated benchmark, not the default compute path. */
    if (npu_queue_create(&p->d, 0, 0, &p->q) != 0) { free(p); return HPI_EDEVICE; }
    p->q_ok = 1;

    dev->priv = p;
    return HPI_OK;
}

static void npu_close(hpi_device *dev) {
    if (!dev || !dev->priv) return;
    npu3720_priv *p = (npu3720_priv *)dev->priv;

    /* Free the staged buffers and blobs we own. The src/ loader table exposes zeMemFree (via
     * npu_mem_free) and zeContextDestroy, but no command-list / queue / graph destroy wrappers, so
     * those handles are reclaimed by process teardown — acceptable because a device is opened once
     * per backend and closed once. (If npu.h later grows destroy wrappers, call them here.) */
    for (int i = 0; i < p->ncache; i++) {
        npu3720_shape *s = &p->cache[i];
        if (s->x_mem) npu_mem_free(&p->d, s->x_mem);
        if (s->y_mem) npu_mem_free(&p->d, s->y_mem);
        free(s->blob);
    }
    if (p->d.ctx && p->z.zeContextDestroy) p->z.zeContextDestroy(p->d.ctx);
    free(p);
    dev->priv = NULL;
}

/* ---------------------------------------------------------------------------------------------- *
 *  Argument staging helpers (pure conversion between host f32 and the blob's device precision).
 * ---------------------------------------------------------------------------------------------- */

/* Copy `n` host floats into device buffer `dst` in the graph-argument precision `prec`. */
static hpi_status stage_in_f32(void *dst, ze_graph_argument_precision_t prec,
                               const float *src, size_t n) {
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
        memcpy(dst, src, n * sizeof(float));
        return HPI_OK;
    }
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
        uint16_t *d16 = (uint16_t *)dst;
        for (size_t i = 0; i < n; i++) d16[i] = npu_f32_to_f16(src[i]);
        return HPI_OK;
    }
    return HPI_EDEVICE;   /* blob wants a precision the plumbing doesn't convert to yet */
}

/* Copy `n` values out of device buffer `src` (precision `prec`) into host floats `dst`. */
static hpi_status stage_out_f32(float *dst, ze_graph_argument_precision_t prec,
                                const void *src, size_t n) {
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
        memcpy(dst, src, n * sizeof(float));
        return HPI_OK;
    }
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
        const uint16_t *s16 = (const uint16_t *)src;
        for (size_t i = 0; i < n; i++) dst[i] = npu_f16_to_f32(s16[i]);
        return HPI_OK;
    }
    return HPI_EDEVICE;
}

/* Build a compiled blob + graph + staged buffers for this shape and register it in the cache.
 * Returns the new entry, or NULL on failure (with *st set). */
static npu3720_shape *shape_build(npu3720_priv *p, const hpi_q8_0_gemm *op, hpi_status *st) {
    if (p->ncache >= NPU3720_CACHE_MAX) { *st = HPI_UNAVAILABLE; return NULL; }  /* raise the cap */

    npu3720_shape tmp;
    memset(&tmp, 0, sizeof tmp);
    tmp.w_key = op->w; tmp.M = op->M; tmp.N = op->N; tmp.K = op->K;

    /* Step 2 (THE SEAM): get a native blob for this shape. Unavailable until built on the box. */
    hpi_status bs = hpi_npu3720_build_blob(op, &tmp.blob, &tmp.blob_len, &tmp.io);
    if (bs != HPI_OK) { *st = bs; return NULL; }

    /* Weights-as-input (strategy B) needs an encoder the plumbing doesn't have yet — reject clearly
     * rather than stage garbage. Strategy A (baked weights, arg_w < 0) is the supported path. */
    if (tmp.io.arg_w >= 0) { free(tmp.blob); *st = HPI_EDEVICE; return NULL; }

    /* Step 4a: load the blob as a graph and read back its argument schema. */
    if (npu_graph_create(&p->d, tmp.blob, tmp.blob_len, &tmp.g) != 0) {
        free(tmp.blob); *st = HPI_EDEVICE; return NULL;
    }

    /* Validate the seam's arg indices against what the graph actually reports. */
    if (tmp.io.arg_x < 0 || tmp.io.arg_y < 0 ||
        (uint32_t)tmp.io.arg_x >= tmp.g.nargs || (uint32_t)tmp.io.arg_y >= tmp.g.nargs ||
        tmp.g.arg[tmp.io.arg_x].is_output || !tmp.g.arg[tmp.io.arg_y].is_output) {
        free(tmp.blob); *st = HPI_EDEVICE; return NULL;
    }

    const npu_graph_arg *ax = &tmp.g.arg[tmp.io.arg_x];
    const npu_graph_arg *ay = &tmp.g.arg[tmp.io.arg_y];

    /* Shapes must match what we're about to compute: X is M*K, Y is M*N. */
    if (ax->elems != (size_t)op->M * (size_t)op->K ||
        ay->elems != (size_t)op->M * (size_t)op->N) {
        free(tmp.blob); *st = HPI_EDEVICE; return NULL;
    }

    /* Step 3: allocate NPU-visible buffers for the input and output arguments and bind them. */
    tmp.x_mem = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, ax->bytes);
    tmp.y_mem = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, ay->bytes);
    if (!tmp.x_mem || !tmp.y_mem) {
        if (tmp.x_mem) npu_mem_free(&p->d, tmp.x_mem);
        if (tmp.y_mem) npu_mem_free(&p->d, tmp.y_mem);
        free(tmp.blob); *st = HPI_ENOMEM; return NULL;
    }
    if (npu_graph_set_arg(&tmp.g, (uint32_t)tmp.io.arg_x, tmp.x_mem) != 0 ||
        npu_graph_set_arg(&tmp.g, (uint32_t)tmp.io.arg_y, tmp.y_mem) != 0) {
        npu_mem_free(&p->d, tmp.x_mem); npu_mem_free(&p->d, tmp.y_mem);
        free(tmp.blob); *st = HPI_EDEVICE; return NULL;
    }

    p->cache[p->ncache] = tmp;
    npu3720_shape *slot = &p->cache[p->ncache];
    p->ncache++;
    *st = HPI_OK;
    return slot;
}

static npu3720_shape *shape_find(npu3720_priv *p, const hpi_q8_0_gemm *op) {
    for (int i = 0; i < p->ncache; i++) {
        npu3720_shape *s = &p->cache[i];
        if (s->w_key == op->w && s->M == op->M && s->N == op->N && s->K == op->K) return s;
    }
    return NULL;
}

/* ---------------------------------------------------------------------------------------------- *
 *  gemm() — one validated Q8_0 GEMM (shapes already checked by the dispatcher).
 * ---------------------------------------------------------------------------------------------- */
static hpi_status npu_gemm(hpi_device *dev, const hpi_q8_0_gemm *op) {
    if (!dev || !dev->priv || !op) return HPI_EINVAL;
    npu3720_priv *p = (npu3720_priv *)dev->priv;

    hpi_status st = HPI_OK;
    npu3720_shape *s = shape_find(p, op);
    if (!s) { s = shape_build(p, op, &st); if (!s) return st; }

    const npu_graph_arg *ax = &s->g.arg[s->io.arg_x];
    const npu_graph_arg *ay = &s->g.arg[s->io.arg_y];

    /* Stage activations X (host f32) into the device input buffer in the blob's precision. */
    st = stage_in_f32(s->x_mem, ax->precision, op->x, (size_t)op->M * (size_t)op->K);
    if (st != HPI_OK) return st;

    /* Step 4b: initialize the graph on the device once (loads baked weights / builds descriptors),
     * then execute. AppendGraphInitialize + Execute is the VPU_CMD_INFERENCE_EXECUTE path. */
    if (!s->initialized) {
        ze_command_list_handle_t li;
        if (npu_list_create(&p->d, &li) != 0) return HPI_EDEVICE;
        if (npu_list_graph_init(&s->g, li) != 0) return HPI_EDEVICE;
        if (npu_queue_run(&p->d, p->q, li, UINT64_MAX) != 0) return HPI_EDEVICE;  /* halt on DEVICE_LOST; caller must not loop-resubmit */
        s->initialized = 1;
    }

    ze_command_list_handle_t le;
    if (npu_list_create(&p->d, &le) != 0) return HPI_EDEVICE;
    if (npu_list_graph_exec(&s->g, le) != 0) return HPI_EDEVICE;
    if (npu_queue_run(&p->d, p->q, le, UINT64_MAX) != 0) return HPI_EDEVICE;

    /* Step 5: read back Y (device precision -> host f32). */
    st = stage_out_f32(op->y, ay->precision, s->y_mem, (size_t)op->M * (size_t)op->N);
    return st;
}

static const hpi_backend_ops g_npu_ops = {
    .kind = HPI_BACKEND_NPU_3720,
    .name = "npu-3720",
    .is_hw = 1,
    .available = npu_available,
    .open = npu_open,
    .gemm = npu_gemm,
    .close = npu_close,
};

const hpi_backend_ops *hpi_backend_npu_3720(void) { return &g_npu_ops; }

#else  /* !HPI_HAVE_NPU_3720 — non-hardware build: no NPU backend at all */

const hpi_backend_ops *hpi_backend_npu_3720(void) { return (const hpi_backend_ops *)0; }

#endif
