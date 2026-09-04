/* Versioned baked-weight cache. See re/blob_cache_format.py for the wire format.
 * Full source SHA256 is checked on cold lookup, payload SHA256 before graph load.
 * Weights and cache directory must remain fixed for the HPI device lifetime.
 * Missing, stale or damaged entries return HPI_UNAVAILABLE for caller fallback.
 */
#include "hpi_backend.h"
#if defined(HPI_HAVE_NPU_3720)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hpi_npu3720_blob.h"
#include "hpi_q8_0.h"
#include "sha256/sha256.h"

#define NPU_CACHE_HEADER 128u
#define NPU_CACHE_VERSION 2u

static uint32_t read32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read64(const unsigned char *p) {
    return (uint64_t)read32(p) | ((uint64_t)read32(p + 4) << 32);
}

static void digest_hex(const unsigned char digest[32], char hex[65]) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; ++i) {
        hex[2 * i] = digits[digest[i] >> 4];
        hex[2 * i + 1] = digits[digest[i] & 15];
    }
    hex[64] = 0;
}

static int read_file(const char *path, unsigned char **buffer, size_t *length) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return -1; }
    const long bytes = ftell(f);
    if (bytes < (long)(NPU_CACHE_HEADER + 4u) || fseek(f, 0, SEEK_SET)) {
        fclose(f); return -1;
    }
    unsigned char *p = (unsigned char *)malloc((size_t)bytes);
    if (!p) { fclose(f); return -1; }
    const size_t got = fread(p, 1, (size_t)bytes, f);
    const int closed = fclose(f);
    if (got != (size_t)bytes || closed) { free(p); return -1; }
    *buffer = p; *length = (size_t)bytes;
    return 0;
}

hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
    uint8_t **blob, size_t *blob_len, hpi_npu3720_blob_io *io) {
    if (!blob || !blob_len || !io) return HPI_EINVAL;
    *blob = NULL; *blob_len = 0;
    if (!op || !op->w || op->M <= 0 || op->N <= 0 || op->K <= 0 ||
        op->K % HPI_QK8_0) return HPI_EINVAL;
    if (op->M > 256) return HPI_UNAVAILABLE;
    const char *cache = getenv("NPU_BLOB_CACHE");
    if (!cache || !cache[0]) return HPI_UNAVAILABLE;
    char path[1024];
    const int marker_name = snprintf(path, sizeof path, "%s/cache-v2.ready", cache);
    if (marker_name <= 0 || (size_t)marker_name >= sizeof path) return HPI_UNAVAILABLE;
    FILE *marker = fopen(path, "rb");
    if (!marker) return HPI_UNAVAILABLE;
    unsigned char format[13];
    const size_t marker_bytes = fread(format, 1, sizeof format, marker);
    const int marker_closed = fclose(marker);
    if (marker_bytes != 12 || marker_closed || memcmp(format, "NPUC3720", 8) ||
        read32(format + 8) != NPU_CACHE_VERSION) return HPI_UNAVAILABLE;
    const uint64_t row_blocks = (uint64_t)op->K / HPI_QK8_0;
    if (row_blocks > SIZE_MAX / sizeof(hpi_block_q8_0)) return HPI_EINVAL;
    const size_t row_bytes = (size_t)row_blocks * sizeof(hpi_block_q8_0);
    if ((uint64_t)op->N > SIZE_MAX / row_bytes) return HPI_EINVAL;
    const size_t wbytes = (size_t)op->N * row_bytes;
    const uint64_t capacity = op->M == 1 ? 1u : 256u;
    unsigned char source_digest[32];
    sha256_hash(source_digest, (const unsigned char *)op->w, wbytes);
    char hex[65];
    digest_hex(source_digest, hex);
    const int written = snprintf(path, sizeof path, "%s/v2_%s_n%lld_k%lld_m%llu.npub",
        cache, hex, (long long)op->N, (long long)op->K, (unsigned long long)capacity);
    if (written <= 0 || (size_t)written >= sizeof path) return HPI_UNAVAILABLE;
    unsigned char *file = NULL;
    size_t length = 0;
    if (read_file(path, &file, &length)) return HPI_UNAVAILABLE;
    /* Offsets are the explicit little-endian 128-byte v2 header, never a C struct cast. */
    const uint32_t slab = read32(file + 48), layout = read32(file + 52);
    const int layout_ok = capacity == 1 ? (layout == 1 || layout == 2) : layout == 3;
    if (memcmp(file, "NPUC3720", 8) || read32(file + 8) != NPU_CACHE_VERSION ||
        read32(file + 12) != NPU_CACHE_HEADER || read32(file + 16) != 8 ||
        read32(file + 20) != 1 || read64(file + 24) != capacity ||
        read64(file + 32) != (uint64_t)op->N || read64(file + 40) != (uint64_t)op->K ||
        !slab || slab % 32 || (uint64_t)op->N % slab || !layout_ok ||
        (layout == 2 && (uint64_t)op->N % (2ull * slab)) ||
        read64(file + 56) != wbytes || memcmp(file + 64, source_digest, 32) ||
        memcmp(file + NPU_CACHE_HEADER, "\177ELF", 4)) {
        free(file); return HPI_UNAVAILABLE;
    }
    unsigned char payload_digest[32];
    const size_t payload_size = length - NPU_CACHE_HEADER;
    sha256_hash(payload_digest, file + NPU_CACHE_HEADER, payload_size);
    if (memcmp(file + 96, payload_digest, 32)) { free(file); return HPI_UNAVAILABLE; }
    /* Reuse the allocation; caller owns the ELF buffer and frees it after graph teardown. */
    memmove(file, file + NPU_CACHE_HEADER, payload_size);
    *blob = file; *blob_len = payload_size;
    io->arg_x = 0; io->arg_y = 1; io->arg_w = -1;
    return HPI_OK;
}
#endif
