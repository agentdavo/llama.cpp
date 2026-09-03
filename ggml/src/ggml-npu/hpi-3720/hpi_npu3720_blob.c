/*
 * hpi_npu3720_blob.c — the compute seam (step 2), the "provider" TU. Implements hpi_npu3720_build_blob
 * by loading a per-tensor blob precompiled offline by re/build_blob_cache.py.
 *
 * Contract: weights are FIXED across all decode steps, so we don't author at runtime — an offline pass
 * authored one baked, LINEAR-slab (SWZ=0) M=1 DPU matmul blob per Q8_0 weight tensor, keyed by an
 * FNV-1a-64 of the tensor's RAW Q8_0 bytes (exactly what ggml mmaps as op->w). Here we hash op->w the
 * same way and load <NPU_BLOB_CACHE>/<key>.blob. Baked weights -> arg_w<0, only input 'x', only output
 * 'Relu_5' (arg_y). If no blob is cached for this op, return HPI_UNAVAILABLE so npu_gemm falls back to
 * the CPU reference (the shape/tensor just isn't accelerated yet — never a wrong result).
 *
 * The hash MUST match re/build_blob_cache.py fnv1a64_np exactly: fnv1a64(head[:H]) ^ (fnv1a64(tail[-H:])
 * * GOLDEN) ^ len, with H = min(len, 4096), GOLDEN = 0x9E3779B97F4A7C15, all in wrapping uint64.
 */
#include "hpi_backend.h"   /* pulls hpi.h; brings HPI_HAVE_NPU_3720 into scope via the build defs */

#if defined(HPI_HAVE_NPU_3720)   /* provider only in the hardware build (paired with HPI_NPU3720_BLOB_READY) */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hpi_npu3720_blob.h"
#include "hpi_q8_0.h"    /* hpi_block_q8_0, HPI_QK8_0 */

#define NPU_FNV64_OFFSET 0xcbf29ce484222325ULL
#define NPU_FNV64_PRIME  0x100000001b3ULL
#define NPU_GOLDEN64     0x9E3779B97F4A7C15ULL
#define NPU_HASH_WINDOW  4096u

static uint64_t npu_fnv1a64(const unsigned char *b, size_t n) {
    uint64_t h = NPU_FNV64_OFFSET;
    for (size_t i = 0; i < n; i++) h = (h ^ (uint64_t)b[i]) * NPU_FNV64_PRIME;   /* wraps in uint64 */
    return h;
}

/* Content key over the raw weight bytes — head window ^ (tail window * golden) ^ length. */
static uint64_t npu_weight_key(const unsigned char *a, size_t n) {
    size_t hw = n < NPU_HASH_WINDOW ? n : NPU_HASH_WINDOW;
    uint64_t hh = npu_fnv1a64(a, hw);
    uint64_t tt = npu_fnv1a64(a + (n - hw), hw);
    return hh ^ (tt * NPU_GOLDEN64) ^ (uint64_t)n;
}

/* Read an entire file into a malloc'd buffer. Returns 0 on success (*buf owned by caller), -1 otherwise. */
static int npu_read_file(const char *path, unsigned char **buf, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
    unsigned char *p = (unsigned char *)malloc((size_t)sz);
    if (!p) { fclose(f); return -1; }
    size_t got = fread(p, 1, (size_t)sz, f);
    fclose(f);
    if (got != (size_t)sz) { free(p); return -1; }
    *buf = p; *len = (size_t)sz;
    return 0;
}

hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                  uint8_t **blob, size_t *blob_len,
                                  hpi_npu3720_blob_io *io) {
    if (!op || !blob || !blob_len || !io) return HPI_EINVAL;

    /* The cached blobs are M=1 (decode) matmuls (emit_m1mm/emit_buildblob). llama's prefill runs these
     * ops with M>1 (many tokens at once); there is no M>1 blob yet, so decline and let npu_gemm fall
     * back to the CPU reference for prefill. Decode (M=1) is where the DPU is used. */
    if (op->M != 1) return HPI_UNAVAILABLE;

    const char *cache = getenv("NPU_BLOB_CACHE");
    if (!cache || !cache[0]) return HPI_UNAVAILABLE;   /* no cache configured -> CPU reference */

    /* Raw weight byte span, exactly as offline: N rows * (K/QK8_0) blocks * sizeof(block). */
    const size_t nblocks = (size_t)op->N * ((size_t)op->K / (size_t)HPI_QK8_0);
    const size_t wbytes  = nblocks * sizeof(hpi_block_q8_0);
    const uint64_t key   = npu_weight_key((const unsigned char *)op->w, wbytes);

    char path[1024];
    int n = snprintf(path, sizeof path, "%s/%016llx.blob", cache, (unsigned long long)key);
    if (n <= 0 || (size_t)n >= sizeof path) return HPI_UNAVAILABLE;

    unsigned char *buf = NULL; size_t len = 0;
    if (npu_read_file(path, &buf, &len) != 0) return HPI_UNAVAILABLE;   /* not cached -> CPU reference */

    { static int said = 0; if (!said) { said = 1;
        fprintf(stderr, "hpi-3720: loaded cached DPU blob (key=%016llx N=%lld K=%lld, %zu B) — this op runs on the DPU.\n",
                (unsigned long long)key, (long long)op->N, (long long)op->K, len); } }
    *blob = buf; *blob_len = len;
    io->arg_x = 0;   /* 'x'      (input,  sym 1) */
    io->arg_y = 1;   /* 'Relu_5' (output, sym 2) */
    io->arg_w = -1;  /* weights baked into the blob */
    return HPI_OK;
}

#endif /* HPI_HAVE_NPU_3720 */
