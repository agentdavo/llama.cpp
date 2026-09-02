/*
 * test_gguf.c — write a minimal valid GGUF containing one Q8_0 weight tensor, read it back with the
 * hpi_gguf reader, and run it through the Q8_0 offload, comparing against an independent double
 * reference. Proves the whole GGUF -> offload path end to end with no network and no real model.
 * Exit 0 = PASS. (Also leaves the synthetic file so gguf_q8_0_offload can be pointed at it.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../hpi.h"
#include "../hpi_gguf.h"

static void w_u32(FILE *f, uint32_t v, size_t *pos) { fwrite(&v, 4, 1, f); *pos += 4; }
static void w_u64(FILE *f, uint64_t v, size_t *pos) { fwrite(&v, 8, 1, f); *pos += 8; }
static void w_str(FILE *f, const char *s, size_t *pos) { uint64_t n = strlen(s); w_u64(f, n, pos); fwrite(s, 1, n, f); *pos += n; }

int main(void) {
    const int64_t K = 256, N = 12;                  /* K multiple of 32 */
    const int64_t kb = K / 32;
    const char *tname = "blk.0.test.weight";
    const char *path  = "hpi_synth_q8_0.gguf";

    /* build + quantize weights */
    float          *wf = (float *)malloc((size_t)(N * K) * sizeof *wf);
    hpi_block_q8_0 *wq = (hpi_block_q8_0 *)malloc((size_t)(N * kb) * sizeof *wq);
    if (!wf || !wq) return 1;
    uint32_t rng = 0x9E3779B9u;
    for (int64_t i = 0; i < N * K; i++) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; wf[i] = ((float)(rng >> 8) / (float)(1u << 24)) * 4.0f - 2.0f; }
    for (int64_t n = 0; n < N; n++) if (hpi_q8_0_quantize_row(wf + n * K, wq + n * kb, K) != 0) return 1;

    /* write a minimal GGUF v3: 1 KV (general.alignment=32), 1 tensor */
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); return 1; }
    size_t pos = 0;
    w_u32(f, 0x46554747u, &pos);        /* "GGUF" */
    w_u32(f, 3, &pos);                  /* version */
    w_u64(f, 1, &pos);                  /* n_tensors */
    w_u64(f, 1, &pos);                  /* n_kv */
    w_str(f, "general.alignment", &pos);
    w_u32(f, 4, &pos);                  /* value type U32 */
    w_u32(f, 32, &pos);                 /* alignment */
    w_str(f, tname, &pos);             /* tensor name */
    w_u32(f, 2, &pos);                  /* n_dims */
    w_u64(f, (uint64_t)K, &pos);        /* ne[0] */
    w_u64(f, (uint64_t)N, &pos);        /* ne[1] */
    w_u32(f, HPI_GGML_TYPE_Q8_0, &pos); /* type */
    w_u64(f, 0, &pos);                  /* offset */
    while (pos % 32) { fputc(0, f); pos++; }         /* pad to alignment */
    fwrite(wq, sizeof *wq, (size_t)(N * kb), f);
    fclose(f);

    /* read it back */
    hpi_gguf g;
    if (hpi_gguf_open(path, &g) != 0) { fprintf(stderr, "open failed\n"); return 1; }
    int ok = (g.n_tensors == 1) && (g.alignment == 32);
    const hpi_gguf_tensor *t = hpi_gguf_find(&g, tname);
    ok &= (t != NULL);
    if (t) ok &= (t->type == HPI_GGML_TYPE_Q8_0 && t->ne[0] == (uint64_t)K && t->ne[1] == (uint64_t)N &&
                  t->nbytes == (uint64_t)(N * kb) * sizeof(hpi_block_q8_0) &&
                  memcmp(t->data, wq, (size_t)(N * kb) * sizeof *wq) == 0);
    printf("  [%s] gguf read: %u tensors, align=%u, tensor '%s' dims [%lld,%lld] type=%d nbytes=%llu\n",
           ok ? "PASS" : "FAIL", g.n_tensors, g.alignment, t ? t->name : "(null)",
           t ? (long long)t->ne[0] : -1, t ? (long long)t->ne[1] : -1, t ? t->type : -1,
           t ? (unsigned long long)t->nbytes : 0ull);
    if (!ok || !t) { hpi_gguf_close(&g); return 1; }

    /* offload a GEMM through the parsed tensor, compare to a double reference */
    const int64_t M = 5;
    float *x = (float *)malloc((size_t)(M * K) * sizeof *x);
    float *y = (float *)malloc((size_t)(M * N) * sizeof *y);
    if (!x || !y) return 1;
    for (int64_t i = 0; i < M * K; i++) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; x[i] = ((float)(rng >> 8) / (float)(1u << 24)) * 2.0f - 1.0f; }

    hpi_device *dev = hpi_open(HPI_BACKEND_AUTO, NULL);
    if (!dev) return 1;
    const hpi_q8_0_gemm op = { M, N, K, (const hpi_block_q8_0 *)(const void *)t->data, x, y };
    if (hpi_q8_0_gemm_run(dev, &op) != HPI_OK) return 1;

    double max_rel = 0.0;
    for (int64_t m = 0; m < M; m++)
        for (int64_t n = 0; n < N; n++) {
            const hpi_block_q8_0 *wr = (const hpi_block_q8_0 *)(const void *)t->data + n * kb;
            double acc = 0.0;
            for (int64_t b = 0; b < kb; b++) {
                const double d = (double)hpi_f16_to_f32(wr[b].d);
                for (int i = 0; i < 32; i++) acc += (double)wr[b].qs[i] * d * (double)x[m * K + b * 32 + i];
            }
            const double diff = fabs((double)y[m * N + n] - acc) / (fabs(acc) + 1e-6);
            if (diff > max_rel) max_rel = diff;
        }
    const int ok2 = max_rel < 1e-4;
    printf("  [%s] gguf-fed offload: max_rel=%.3e\n", ok2 ? "PASS" : "FAIL", max_rel);

    hpi_close(dev); hpi_gguf_close(&g);
    free(wf); free(wq); free(x); free(y);
    printf("%s\n", (ok && ok2) ? "ALL PASS" : "FAILURES");
    return (ok && ok2) ? 0 : 1;
}
