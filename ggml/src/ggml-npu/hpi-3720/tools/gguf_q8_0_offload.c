/*
 * gguf_q8_0_offload.c — point the hpi-3720 Q8_0 offload at a real .gguf model.
 *
 * Reads a GGUF, finds a 2-D Q8_0 weight tensor (named, or the first one), synthesizes an M-row fp32
 * activation, runs it through hpi_q8_0_gemm_run, and reports timing + an output checksum. With
 * --check it also compares against an independent double-precision dequant reference.
 *
 * Usage: gguf_q8_0_offload <model.gguf> [--tensor <name>] [--rows M] [--check] [--list]
 *
 * This is the harness that will drive Qwen3.5-0.8B-Q8_0.gguf once it is on a machine that can reach
 * Hugging Face (the sandbox's egress blocks it). It needs no NPU: today the offload resolves to the
 * hpi CPU reference; on a hardware build it dispatches to the VPU-3720 unchanged.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "hpi.h"
#include "hpi_gguf.h"

/* portable coarse timer (CPU time). Good enough for a GFLOP/s estimate; no POSIX/Win split. */
static double now_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

static const hpi_gguf_tensor *pick_q8_0(const hpi_gguf *g, const char *want) {
    if (want) {
        const hpi_gguf_tensor *t = hpi_gguf_find(g, want);
        if (!t) { fprintf(stderr, "tensor '%s' not found\n", want); return NULL; }
        if (t->type != HPI_GGML_TYPE_Q8_0) { fprintf(stderr, "tensor '%s' is not Q8_0 (type=%d)\n", want, t->type); return NULL; }
        return t;
    }
    for (uint32_t i = 0; i < g->n_tensors; i++) {
        const hpi_gguf_tensor *t = &g->tensors[i];
        if (t->type == HPI_GGML_TYPE_Q8_0 && t->n_dims == 2 && t->ne[0] % 32 == 0) return t;
    }
    fprintf(stderr, "no 2-D Q8_0 tensor found\n");
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gguf_q8_0_offload <model.gguf> [--tensor <name>] [--rows M] [--check] [--list]\n"); return 2; }
    const char *path = argv[1], *want = NULL;
    int64_t M = 8; int do_check = 0, do_list = 0;
    for (int i = 2; i < argc; i++) {
        if      (!strcmp(argv[i], "--tensor") && i + 1 < argc) want = argv[++i];
        else if (!strcmp(argv[i], "--rows")   && i + 1 < argc) M = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--check"))                  do_check = 1;
        else if (!strcmp(argv[i], "--list"))                   do_list = 1;
    }
    if (M < 1) M = 1;

    hpi_gguf g;
    if (hpi_gguf_open(path, &g) != 0) return 1;
    printf("gguf: %s — %u tensors, alignment %u\n", path, g.n_tensors, g.alignment);

    if (do_list) {
        uint32_t nq = 0;
        for (uint32_t i = 0; i < g.n_tensors; i++) {
            const hpi_gguf_tensor *t = &g.tensors[i];
            if (t->type == HPI_GGML_TYPE_Q8_0) {
                printf("  Q8_0 %-40s [%lld x %lld x %lld x %lld] %llu B\n", t->name,
                       (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                       (unsigned long long)t->nbytes);
                nq++;
            }
        }
        printf("  (%u Q8_0 tensors)\n", nq);
    }

    const hpi_gguf_tensor *w = pick_q8_0(&g, want);
    if (!w) { hpi_gguf_close(&g); return 1; }
    const int64_t K = (int64_t)w->ne[0], N = (int64_t)w->ne[1];
    printf("offloading Q8_0 mul_mat: weights '%s' [K=%lld, N=%lld], activations [M=%lld, K=%lld]\n",
           w->name, (long long)K, (long long)N, (long long)M, (long long)K);

    float *x = (float *)malloc((size_t)(M * K) * sizeof *x);
    float *y = (float *)malloc((size_t)(M * N) * sizeof *y);
    if (!x || !y) { fprintf(stderr, "oom\n"); hpi_gguf_close(&g); return 1; }
    uint32_t rng = 0xC0FFEEu;
    for (int64_t i = 0; i < M * K; i++) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; x[i] = ((float)(rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; }

    hpi_status st = HPI_OK;
    hpi_device *dev = hpi_open(HPI_BACKEND_AUTO, &st);
    if (!dev) { fprintf(stderr, "hpi_open: %s\n", hpi_status_str(st)); hpi_gguf_close(&g); return 1; }
    hpi_device_info info; hpi_get_info(dev, &info);
    printf("hpi backend: %s (hardware=%d)\n", info.name, info.is_hw);

    const hpi_q8_0_gemm op = { M, N, K, (const hpi_block_q8_0 *)(const void *)w->data, x, y };
    double t0 = now_ms();
    st = hpi_q8_0_gemm_run(dev, &op);
    double ms = now_ms() - t0;
    if (st != HPI_OK) { fprintf(stderr, "gemm: %s\n", hpi_status_str(st)); hpi_close(dev); hpi_gguf_close(&g); return 1; }

    double sum = 0.0; for (int64_t i = 0; i < M * N; i++) sum += y[i];
    const double gflop = 2.0 * (double)M * (double)N * (double)K / 1e9;
    printf("done: %.2f ms  (%.2f GFLOP, %.1f GFLOP/s)  checksum=%.6f\n", ms, gflop, gflop / (ms / 1e3), sum);

    int rc = 0;
    if (do_check) {
        const int64_t kb = K / 32;
        double max_rel = 0.0;
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++) {
                const hpi_block_q8_0 *wr = (const hpi_block_q8_0 *)(const void *)w->data + n * kb;
                double acc = 0.0;
                for (int64_t b = 0; b < kb; b++) {
                    const double d = (double)hpi_f16_to_f32(wr[b].d);
                    for (int i = 0; i < 32; i++) acc += (double)wr[b].qs[i] * d * (double)x[m * K + b * 32 + i];
                }
                const double diff = fabs((double)y[m * N + n] - acc) / (fabs(acc) + 1e-6);
                if (diff > max_rel) max_rel = diff;
            }
        rc = (max_rel < 1e-4) ? 0 : 1;
        printf("check: max_rel=%.3e  [%s]\n", max_rel, rc == 0 ? "PASS" : "FAIL");
    }

    free(x); free(y); hpi_close(dev); hpi_gguf_close(&g);
    return rc;
}
