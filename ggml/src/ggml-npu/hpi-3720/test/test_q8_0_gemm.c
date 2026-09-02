/*
 * test_q8_0_gemm.c — deterministic self-test for the HPI Q8_0 GEMM (src/ verify-twice discipline:
 * fixed seed, fixed shapes, a reference computed independently of the code under test).
 *
 * It quantizes a known weight matrix to Q8_0, runs the GEMM through the HPI (CPU backend), and
 * compares against a naive double-precision reference over the SAME quantized weights (so the only
 * differences are float vs double accumulation, well within tolerance). Also checks that argument
 * validation rejects malformed ops. Exit 0 = PASS.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../hpi.h"

/* tiny deterministic PRNG (xorshift32) — no libc rand() dependence, reproducible everywhere */
static uint32_t s_rng = 0x1234567u;
static float frand(void) {
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 17; s_rng ^= s_rng << 5;
    return ((float)(s_rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f;   /* [-1, 1) */
}

/* reference GEMM straight from the quantized weights, accumulated in double */
static void ref_gemm(int64_t M, int64_t N, int64_t K,
                     const hpi_block_q8_0 *w, const float *x, double *y) {
    const int64_t kb = K / HPI_QK8_0;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            const hpi_block_q8_0 *wr = w + n * kb;
            const float *xr = x + m * K;
            double acc = 0.0;
            for (int64_t b = 0; b < kb; b++) {
                const double d = (double)hpi_f16_to_f32(wr[b].d);
                for (int i = 0; i < HPI_QK8_0; i++)
                    acc += (double)wr[b].qs[i] * d * (double)xr[b * HPI_QK8_0 + i];
            }
            y[m * N + n] = acc;
        }
}

static int run_case(int64_t M, int64_t N, int64_t K) {
    const int64_t kb = K / HPI_QK8_0;
    float          *wf = (float *)malloc((size_t)(N * K) * sizeof *wf);
    hpi_block_q8_0 *w  = (hpi_block_q8_0 *)malloc((size_t)(N * kb) * sizeof *w);
    float          *x  = (float *)malloc((size_t)(M * K) * sizeof *x);
    float          *y  = (float *)malloc((size_t)(M * N) * sizeof *y);
    double         *yr = (double *)malloc((size_t)(M * N) * sizeof *yr);
    if (!wf || !w || !x || !y || !yr) { fprintf(stderr, "oom\n"); return 1; }

    for (int64_t i = 0; i < N * K; i++) wf[i] = frand() * 3.0f;
    for (int64_t n = 0; n < N; n++)
        if (hpi_q8_0_quantize_row(wf + n * K, w + n * kb, K) != 0) { fprintf(stderr, "quantize failed\n"); return 1; }
    for (int64_t i = 0; i < M * K; i++) x[i] = frand();

    hpi_status st = HPI_OK;
    hpi_device *dev = hpi_open(HPI_BACKEND_CPU, &st);
    if (!dev) { fprintf(stderr, "hpi_open: %s\n", hpi_status_str(st)); return 1; }

    const hpi_q8_0_gemm op = { .M = M, .N = N, .K = K, .w = w, .x = x, .y = y };
    st = hpi_q8_0_gemm_run(dev, &op);
    if (st != HPI_OK) { fprintf(stderr, "gemm: %s\n", hpi_status_str(st)); hpi_close(dev); return 1; }

    ref_gemm(M, N, K, w, x, yr);

    double max_abs = 0.0, max_rel = 0.0;
    for (int64_t i = 0; i < M * N; i++) {
        const double diff = fabs((double)y[i] - yr[i]);
        const double den  = fabs(yr[i]) + 1e-6;
        if (diff > max_abs) max_abs = diff;
        if (diff / den > max_rel) max_rel = diff / den;
    }
    hpi_close(dev);
    free(wf); free(w); free(x); free(y); free(yr);

    const int ok = (max_abs < 1e-2) && (max_rel < 1e-4);
    printf("  [%s] M=%lld N=%lld K=%lld  max_abs=%.3e max_rel=%.3e\n",
           ok ? "PASS" : "FAIL", (long long)M, (long long)N, (long long)K, max_abs, max_rel);
    return ok ? 0 : 1;
}

static int run_validation(void) {
    hpi_device *dev = hpi_open(HPI_BACKEND_CPU, NULL);
    if (!dev) { fprintf(stderr, "hpi_open(CPU) failed\n"); return 1; }
    hpi_block_q8_0 w = {0}; float x = 0, y = 0;
    const hpi_q8_0_gemm bad_k   = { .M=1, .N=1, .K=7,  .w=&w, .x=&x, .y=&y }; /* K not block-aligned */
    const hpi_q8_0_gemm bad_nul = { .M=1, .N=1, .K=32, .w=NULL, .x=&x, .y=&y };
    int ok = 1;
    ok &= (hpi_q8_0_gemm_run(dev, &bad_k)   == HPI_EINVAL);
    ok &= (hpi_q8_0_gemm_run(dev, &bad_nul) == HPI_EINVAL);
    ok &= (hpi_q8_0_gemm_run(dev, NULL)     == HPI_EINVAL);
    printf("  [%s] argument validation rejects malformed ops\n", ok ? "PASS" : "FAIL");
    hpi_close(dev);
    return ok ? 0 : 1;
}

int main(void) {
    hpi_device_info info; hpi_device *d = hpi_open(HPI_BACKEND_AUTO, NULL);
    if (d && hpi_get_info(d, &info) == HPI_OK)
        printf("hpi: backend=%s hw=%d (npu-3720 available=%d)\n",
               info.name, info.is_hw, hpi_backend_available(HPI_BACKEND_NPU_3720));
    hpi_close(d);

    int rc = 0;
    rc |= run_case(4, 8, 32);
    rc |= run_case(3, 5, 64);
    rc |= run_case(16, 16, 256);
    rc |= run_case(1, 1, 4096);
    rc |= run_validation();
    printf("%s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
}
