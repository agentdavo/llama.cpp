/*
 * hpi_npu3720_repack.h - repack a Q8_0 weight into the per-channel-i8 SWZ=0 slab image the resident
 * DPU GEMM reads (the ggml-hexagon repack pattern for hpi-3720). Drop-in for hpi-3720/.
 *
 * Layout (matches emit_buildblob / emit_m1mm slab_image, SWZ=0, proven bit-exact on silicon):
 *   per output channel n:  sw[n] = max|dequant(W[n])| / 127   (amax 0 -> 1)
 *                          Wq[n][k] = clip(round_nearest_even(dequant(W[n][k]) / sw[n]), -127, 127)
 *   per slab of `slabch` channels:  [ table: slabch x 16B {u32 data_off, u32 0x00ffffff, f32 sw, u32 0} ]
 *                                   [ rows:  slabch x K int8, row-major ]
 *   data_off is the OFFSET of the channel's row from the slab base (slabch*16 + z*K). The caller
 *   relocates it to the resident CMX/host address of that slab.
 * out size = (size_t)N*16 + (size_t)N*K bytes. Returns 0, or -1 on bad shape.
 */
#ifndef HPI_NPU3720_REPACK_H
#define HPI_NPU3720_REPACK_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "hpi_q8_0.h"

static inline int hpi_repack_q8_0(const hpi_block_q8_0 *w, int N, int K, int slabch, uint8_t *out) {
    if (slabch <= 0 || K % HPI_QK8_0 != 0 || N % slabch != 0) return -1;
    const int    nb          = K / HPI_QK8_0;           /* Q8_0 blocks per row */
    const int    nslab       = N / slabch;
    const size_t table_bytes = (size_t) slabch * 16;
    const size_t slab_bytes  = table_bytes + (size_t) slabch * (size_t) K;

    for (int s = 0; s < nslab; s++) {
        uint8_t *sl   = out + (size_t) s * slab_bytes;
        int8_t  *rows = (int8_t *) (sl + table_bytes);
        for (int z = 0; z < slabch; z++) {
            const int n = s * slabch + z;
            const hpi_block_q8_0 *row = w + (size_t) n * nb;

            float amax = 0.0f;
            for (int b = 0; b < nb; b++) {
                const float d = hpi_f16_to_f32(row[b].d);
                for (int i = 0; i < HPI_QK8_0; i++) {
                    const float a = fabsf((float) row[b].qs[i] * d);
                    if (a > amax) amax = a;
                }
            }
            if (amax == 0.0f) amax = 1.0f;
            const float sw = amax / 127.0f;

            uint8_t *e = sl + (size_t) z * 16;
            const uint32_t off = (uint32_t) (table_bytes + (size_t) z * (size_t) K);
            const uint32_t magic = 0x00ffffffu, zero = 0;
            memcpy(e + 0, &off, 4); memcpy(e + 4, &magic, 4); memcpy(e + 8, &sw, 4); memcpy(e + 12, &zero, 4);

            int8_t *dst = rows + (size_t) z * (size_t) K;
            for (int b = 0; b < nb; b++) {
                const float d = hpi_f16_to_f32(row[b].d);
                for (int i = 0; i < HPI_QK8_0; i++) {
                    const float v = ((float) row[b].qs[i] * d) / sw;
                    long q = lrintf(v);                 /* round-half-to-even (numpy default) */
                    if (q > 127) q = 127; else if (q < -127) q = -127;
                    dst[b * HPI_QK8_0 + i] = (int8_t) q;
                }
            }
        }
    }
    return 0;
}

#endif /* HPI_NPU3720_REPACK_H */
