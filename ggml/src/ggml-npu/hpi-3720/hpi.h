/*
 * hpi.h — Host-Platform Interface to the Intel NPU 2.7 / VPU 3720 ("AI Boost", PCI 8086:AD1D).
 *
 * This is the thin C ABI the ggml-npu host backend calls to offload a Q8_0 matmul. It abstracts
 * over WHO computes:
 *
 *   HPI_BACKEND_CPU       portable C reference — always available, the "cross platform for now" path.
 *   HPI_BACKEND_NPU_3720  the real NPU, driven through the src/ direct-access toolkit (src/npu.h:
 *                         Level Zero -> VPU_CMD_INFERENCE_EXECUTE -> NCE/DPU MAC array). Only built
 *                         with real behavior on Windows x64 with the NPU present (HPI_HAVE_NPU_3720);
 *                         elsewhere it reports HPI_UNAVAILABLE so callers fall back to CPU.
 *
 * Design follows the src/ Carmack discipline: state is explicit (the caller owns an hpi_device and
 * passes it in — no globals), every entry point returns a checked status, and the pure math (the
 * GEMM, the Q8_0 (de)quant) is separated from the effectful device path.
 *
 * The CPU reference is always available. The optional Windows hardware path uses checked
 * Level Zero execution and a versioned blob cache. Missing blobs remain unavailable. See README.md.
 */
#ifndef HPI_H
#define HPI_H

#include <stdint.h>
#include <stddef.h>
/* Portable FP16 conversion and the byte-exact Q8_0 block/row helpers. */

/* uint16_t IEEE-754 binary16 -> float. Matches ggml_half semantics (ggml-common.h). */
static inline float hpi_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                         /* +/- zero */
        } else {
            /* subnormal half -> normalized float */
            uint32_t e = 0, m = mant;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    /* type-pun through memcpy, never an incompatible pointer cast (src/npu.h rule) */
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

/* float -> uint16_t IEEE-754 binary16, round-to-nearest-even, saturating to +/- inf. */
static inline uint16_t hpi_f32_to_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {                 /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {                                  /* overflow -> inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {                                     /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;          /* too small -> +/- 0 */
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t round = 1u << (shift - 1);
        uint32_t v = mant + round;
        if ((mant & ((round << 1) - 1)) == round) v = mant + round - ((v >> shift) & 1u); /* ties-to-even */
        return (uint16_t)(sign | (v >> shift));
    }
    /* normal: round mantissa from 23 to 10 bits, ties-to-even */
    const uint32_t round = 0x1000u;                     /* half of the dropped 13-bit field */
    uint32_t v = mant + round + ((mant >> 13) & 1u);
    uint32_t out_exp  = (uint32_t)exp;
    uint32_t out_mant = v >> 13;
    if (out_mant & 0x400u) { out_mant = 0; out_exp++; } /* mantissa carry */
    if (out_exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (out_exp << 10) | (out_mant & 0x3FFu));
}

#include <assert.h>
#include <math.h>

#define HPI_QK8_0 32   /* elements per Q8_0 block — ggml QK8_0 */

typedef struct {
    uint16_t d;              /* fp16 scale (ggml_half) — delta */
    int8_t   qs[HPI_QK8_0];  /* signed 8-bit quants */
} hpi_block_q8_0;

/* ABI guard: the whole point of mirroring ggml is that this stays true. */
static_assert(sizeof(hpi_block_q8_0) == sizeof(uint16_t) + HPI_QK8_0, "hpi_block_q8_0 must be 34 bytes, matching ggml block_q8_0");

/* Dequantize one block to `HPI_QK8_0` floats. Pure; no state. */
static inline void hpi_q8_0_dequant_block(const hpi_block_q8_0 *b, float *out) {
    const float d = hpi_f16_to_f32(b->d);
    for (int i = 0; i < HPI_QK8_0; i++) out[i] = (float)b->qs[i] * d;
}

/*
 * Quantize one row of `k` floats (k must be a multiple of HPI_QK8_0) into `k / HPI_QK8_0` blocks.
 * Matches ggml quantize_row_q8_0: per block d = max(|x|)/127, qs[i] = round(x[i]/d) (nearest).
 * Returns 0 on success, -1 if k is not block-aligned.
 */
static inline int hpi_q8_0_quantize_row(const float *x, hpi_block_q8_0 *blocks, int64_t k) {
    if (k % HPI_QK8_0 != 0) return -1;
    const int64_t nb = k / HPI_QK8_0;
    for (int64_t b = 0; b < nb; b++) {
        const float *xb = x + b * HPI_QK8_0;
        float amax = 0.0f;
        for (int i = 0; i < HPI_QK8_0; i++) {
            const float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        const float d  = amax / 127.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
        blocks[b].d = hpi_f32_to_f16(d);
        for (int i = 0; i < HPI_QK8_0; i++) {
            const float v = xb[i] * id;
            /* round-half-away-from-zero, as ggml's roundf-based path does */
            blocks[b].qs[i] = (int8_t)lroundf(v);
        }
    }
    return 0;
}


#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HPI_OK               =  0,
    HPI_UNAVAILABLE      = -1,  /* backend not present on this platform/build */
    HPI_EINVAL           = -2,  /* bad argument (null, shape, alignment) */
    HPI_ENOMEM           = -3,  /* allocation failed */
    HPI_EDEVICE          = -4   /* the accelerator or its driver rejected the job */
} hpi_status;

typedef enum {
    HPI_BACKEND_AUTO     = 0,   /* pick NPU_3720 if usable, else CPU */
    HPI_BACKEND_CPU      = 1,
    HPI_BACKEND_NPU_3720 = 2
} hpi_backend_kind;

/* Opaque device/context handle. One owner; created and destroyed explicitly. */
typedef struct hpi_device hpi_device;

typedef struct {
    hpi_backend_kind kind;      /* which backend actually backs this device */
    const char      *name;      /* human-readable, e.g. "cpu-reference" / "npu-3720" */
    int              is_hw;     /* 1 if a real accelerator, 0 if a software reference */
} hpi_device_info;

/* Cumulative API wall times, enabled by GGML_NPU_PROFILE=1 at device open. */
typedef struct {
    int enabled;
    double prepare_ms, input_ms, submit_ms, sync_ms, output_ms, fallback_ms;
    uint64_t submissions, graphs, fallback_ops, cache_misses;
    /* Components of prepare_ms, not additional wall time. init_ms includes
     * the one-time graph initialize submission/wait; sync_ms is execute wait.
     * No API wall counter is a DMA/SHAVE/DPU execution measurement. */
    double lookup_ms, build_ms, init_ms, record_ms;
    uint64_t warm_hits, negative_hits, shape_builds, command_records, list_reuses;
    double host_work_ms; /* CPU callback scope, also included in caller CPU timing. */
    uint64_t host_work_calls;
} hpi_profile;

hpi_status hpi_get_profile(const hpi_device *dev, hpi_profile *out);

/*
 * Q8_0 GEMM offload — ggml mul_mat semantics.
 *
 *   weights W : N rows x K cols, Q8_0-quantized. Row-major blocks: W[n] is (K/HPI_QK8_0) contiguous
 *               hpi_block_q8_0, so `w` points at N*(K/HPI_QK8_0) blocks.
 *   input   X : M rows x K cols, float32, row-major (`x` = M*K floats).
 *   output  Y : M rows x N cols, float32, row-major (`y` = M*N floats), Y[m,n] = sum_k deq(W[n,k])*X[m,k].
 *
 * K must be a multiple of HPI_QK8_0. All buffers are caller-owned host memory; the backend copies
 * in/out as needed (the NPU path will stage them into NPU-visible memory).
 */
typedef struct {
    int64_t                 M, N, K;
    const hpi_block_q8_0   *w;  /* N * (K/HPI_QK8_0) blocks */
    const float            *x;  /* M * K floats */
    float                  *y;  /* M * N floats (output) */
} hpi_q8_0_gemm;

/* Is a given backend usable on this build/platform right now? (Cheap; no device open.) */
int         hpi_backend_available(hpi_backend_kind kind);

/* Open a device for the given backend (AUTO selects). Returns NULL and sets *status on failure. */
hpi_device *hpi_open(hpi_backend_kind kind, hpi_status *status);

/* Query which backend a device ended up using. */
hpi_status  hpi_get_info(const hpi_device *dev, hpi_device_info *out);

/* Run one Q8_0 GEMM. Validates shapes, then dispatches to the device's backend. */
hpi_status  hpi_q8_0_gemm_run(hpi_device *dev, const hpi_q8_0_gemm *op);

/* Run N INDEPENDENT Q8_0 GEMMs as a batch. On the NPU this stages all inputs, submits all N graphs in
 * ONE queue execute + ONE synchronize, then reads all outputs — amortizing the per-op submit+sync that
 * dominates small ops. The ops MUST be independent (none reads another's output). Validates each op;
 * falls back to running them one at a time if the backend has no batch path. */
hpi_status  hpi_q8_0_gemm_batch(hpi_device *dev, const hpi_q8_0_gemm *ops, int n);

/* Execute available hardware ops; per-op HPI_UNAVAILABLE leaves y untouched for caller fallback. */
hpi_status  hpi_q8_0_gemm_batch_try(hpi_device *dev, const hpi_q8_0_gemm *ops, int n, hpi_status *results);

/* Run independent host work once, after the last device submission and before its
 * completion wait (or synchronously if no hardware op is available). The callback
 * must not access any op's output, modify its inputs, or reenter this HPI device.
 * All buffers remain caller-owned until this synchronous function returns. On an
 * earlier device error the callback may not run. NULL work is equivalent to try. */
typedef void (*hpi_host_work)(void *user);
hpi_status hpi_q8_0_gemm_batch_overlap(hpi_device *dev, const hpi_q8_0_gemm *ops,
        int n, hpi_status *results, hpi_host_work work, void *user);

/* Release a device. Safe on NULL. */
void        hpi_close(hpi_device *dev);

/* Human-readable status string (never NULL). */
const char *hpi_status_str(hpi_status s);

#ifdef __cplusplus
}
#endif

#endif /* HPI_H */
