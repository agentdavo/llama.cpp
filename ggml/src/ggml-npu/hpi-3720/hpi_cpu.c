/*
 * hpi_cpu.c — portable CPU reference backend for the Q8_0 GEMM. This is the "cross platform for now"
 * path and the golden reference the NPU backend must match. Plain C11, no intrinsics, no CPU-feature
 * flags — correctness over speed. It dequantizes each weight block once per (n, k-block) and
 * accumulates in float, exactly the arithmetic the NPU path will approximate on the DPU MAC array.
 */
#include <stdlib.h>
#include "hpi_backend.h"

static int cpu_available(void) { return 1; }   /* always */

static hpi_status cpu_open(hpi_device *dev) { (void)dev; return HPI_OK; }
static void       cpu_close(hpi_device *dev) { (void)dev; }

static hpi_status cpu_gemm(hpi_device *dev, const hpi_q8_0_gemm *op) {
    (void)dev;
    const int64_t M = op->M, N = op->N, K = op->K;
    const int64_t kb = K / HPI_QK8_0;             /* blocks per row */

    for (int64_t m = 0; m < M; m++) {
        const float *xr = op->x + m * K;
        float       *yr = op->y + m * N;
        for (int64_t n = 0; n < N; n++) {
            const hpi_block_q8_0 *wr = op->w + n * kb;
            float acc = 0.0f;
            for (int64_t b = 0; b < kb; b++) {
                const float d = hpi_f16_to_f32(wr[b].d);
                const float *xk = xr + b * HPI_QK8_0;
                float bacc = 0.0f;
                for (int i = 0; i < HPI_QK8_0; i++) {
                    bacc += (float)wr[b].qs[i] * xk[i];
                }
                acc += bacc * d;                  /* scale per block, like ggml vec_dot_q8_0 */
            }
            yr[n] = acc;
        }
    }
    return HPI_OK;
}

static const hpi_backend_ops g_cpu_ops = {
    .kind = HPI_BACKEND_CPU,
    .name = "cpu-reference",
    .is_hw = 0,
    .available = cpu_available,
    .open = cpu_open,
    .gemm = cpu_gemm,
    .close = cpu_close,
};

const hpi_backend_ops *hpi_backend_cpu(void) { return &g_cpu_ops; }
