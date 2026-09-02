/*
 * hpi_q8_0.h — the Q8_0 block, mirrored byte-exact from ggml, plus a portable row quantizer.
 *
 * The offload op is a GEMM whose weight operand is Q8_0-quantized. This header defines the block
 * layout the HPI moves to/from the accelerator and the quantizer used by tests and (later) by the
 * host backend when it needs to quantize activations. It must stay bit-identical to ggml's
 * block_q8_0 (ggml/src/ggml-common.h): { ggml_half d; int8_t qs[32]; }, 34 bytes.
 */
#ifndef HPI_Q8_0_H
#define HPI_Q8_0_H

#include <stdint.h>
#include <assert.h>
#include <math.h>
#include "hpi_fp16.h"

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

#endif /* HPI_Q8_0_H */
