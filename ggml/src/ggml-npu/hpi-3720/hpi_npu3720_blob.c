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
#include "hpi_npu_internal.h"
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

static int format_present(const char *cache, unsigned version) {
    char path[1024];
    const int n = snprintf(path, sizeof path, "%s/cache-v%u.ready", cache, version);
    if (n <= 0 || (size_t)n >= sizeof path) return 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned char marker[13];
    const size_t size = fread(marker, 1, sizeof marker, f);
    const int closed = fclose(f);
    return size == 12 && !closed && !memcmp(marker, "NPUC3720", 8) && read32(marker + 8) == version;
}

static hpi_status read_weight_input(const char *cache, const char *source_hex,
    const unsigned char source_digest[32], const hpi_q8_0_gemm *op, size_t wbytes,
    uint64_t capacity, uint8_t **blob, size_t *blob_len, hpi_npu3720_blob_io *io) {
    char path[1024];
    int n = snprintf(path, sizeof path, "%s/v3_%s_n%lld_k%lld_m%llu.npui", cache, source_hex,
                     (long long)op->N, (long long)op->K, (unsigned long long)capacity);
    if (n <= 0 || (size_t)n >= sizeof path) return HPI_UNAVAILABLE;
    unsigned char *image = NULL;
    size_t length = 0;
    if (read_file(path, &image, &length)) return HPI_UNAVAILABLE;
    if (length < 160 || op->K > 8192 ||
        (uint64_t)op->N > (SIZE_MAX - 160u) / ((uint64_t)op->K + 16u)) {
        free(image); return HPI_UNAVAILABLE;
    }
    const size_t image_bytes = (size_t)op->N * ((size_t)op->K + 16u);
    const uint32_t slab = read32(image + 48);
    if (length - 160u != image_bytes || memcmp(image, "NPUC3720", 8) ||
        read32(image + 8) != 3 || read32(image + 12) != 160 || read32(image + 16) != 8 ||
        read32(image + 20) != 1 || read64(image + 24) != capacity ||
        read64(image + 32) != (uint64_t)op->N || read64(image + 40) != (uint64_t)op->K ||
        !slab || slab % 32 || (uint64_t)op->N % (2ull * slab) || read32(image + 52) != 4 ||
        read64(image + 56) != wbytes || memcmp(image + 64, source_digest, 32)) {
        free(image); return HPI_UNAVAILABLE;
    }
    unsigned char digest[32];
    sha256_hash(digest, image + 160, image_bytes);
    if (memcmp(digest, image + 96, 32)) { free(image); return HPI_UNAVAILABLE; }
    char program_hex[65];
    digest_hex(image + 128, program_hex);
    n = snprintf(path, sizeof path, "%s/programs/%s.blob", cache, program_hex);
    if (n <= 0 || (size_t)n >= sizeof path) { free(image); return HPI_UNAVAILABLE; }
    unsigned char *program = NULL;
    size_t program_bytes = 0;
    if (read_file(path, &program, &program_bytes)) { free(image); return HPI_UNAVAILABLE; }
    sha256_hash(digest, program, program_bytes);
    if (memcmp(digest, image + 128, 32) || memcmp(program, "\177ELF", 4)) {
        free(program); free(image); return HPI_UNAVAILABLE;
    }
    memmove(image, image + 160, image_bytes);
    *blob = program; *blob_len = program_bytes;
    io->arg_x = 0; io->arg_w = 1; io->arg_y = 2;
    io->weight_image = image; io->weight_image_bytes = image_bytes;
    io->capacity = (int64_t)capacity;
    return HPI_OK;
}

hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
    uint8_t **blob, size_t *blob_len, hpi_npu3720_blob_io *io) {
    if (!blob || !blob_len || !io) return HPI_EINVAL;
    *blob = NULL; *blob_len = 0;
    memset(io, 0, sizeof *io);
    if (!op || !op->w || op->M <= 0 || op->N <= 0 || op->K <= 0 ||
        op->K % HPI_QK8_0) return HPI_EINVAL;
    if (op->M > 256) return HPI_UNAVAILABLE;
    const char *cache = getenv("NPU_BLOB_CACHE");
    if (!cache || !cache[0]) return HPI_UNAVAILABLE;
    const int has_v2 = format_present(cache, 2);
    const int has_v3 = op->M <= 8 && format_present(cache, 3);
    if (!has_v2 && !has_v3) return HPI_UNAVAILABLE;
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
    /* v3: the 1-row program for plain decode when the cache has it, else (and for M in 2..8, e.g. the
     * MTP verification batch) the 8-row program; the runtime zero-pads unused rows. */
    if (has_v3 && op->M == 1 &&
        read_weight_input(cache, hex, source_digest, op, wbytes, 1, blob, blob_len, io) == HPI_OK) return HPI_OK;
    if (has_v3 && read_weight_input(cache, hex, source_digest, op, wbytes, 8, blob, blob_len, io) == HPI_OK) return HPI_OK;
    if (!has_v2 || (op->M > 1 && op->M <= 8 && op->K != 1024 && op->K != 2048)) return HPI_UNAVAILABLE;
    char path[1024];
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
    io->capacity = (int64_t)capacity;
    return HPI_OK;
}
#endif
