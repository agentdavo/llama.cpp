/* Pure NPU data preparation, residency policy, and blob-provider contract.
 * No device API dependency: ownership/effects remain in the runtime and loader. */
#ifndef HPI_NPU_INTERNAL_H
#define HPI_NPU_INTERNAL_H

#include "hpi.h"   /* hpi_status, hpi_q8_0_gemm */

/* How build_blob's blob binds its runtime graph arguments (see hpi_npu_3720.c shape_build). */
typedef struct {
    int arg_x;   /* graph argument index that receives activations X (>= 0)      */
    int arg_y;   /* graph argument index that produces output Y (>= 0)           */
    int arg_w;   /* weights input index, or < 0 if weights are baked into the blob */
    uint8_t *weight_image;       /* optional malloc-owned immutable image; caller frees */
    size_t weight_image_bytes;
} hpi_npu3720_blob_io;

/* Produce (or fetch from a cache) a native VPU-3720 blob computing Y[MxN] = X[MxK] . dequant(W)^T for
 * this exact shape+weights. On success: *blob = malloc'd blob (caller frees), *blob_len its size, *io
 * filled. Returns HPI_UNAVAILABLE when no blob is available for this op (caller falls back to CPU).
 * Declared only when a provider TU (hpi_npu3720_blob.c) is compiled in; otherwise hpi_npu_3720.c
 * supplies a local static stub. */
#if defined(HPI_NPU3720_BLOB_READY)
hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                  uint8_t **blob, size_t *blob_len,
                                  hpi_npu3720_blob_io *io);
#endif

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "hpi.h"

/* Q8_0 -> per-channel-i8 SWZ=0 slab image, matching emit_m1mm slab_image.
 * Per channel: scale=max(abs(dequant(W)))/127 (zero -> 1); quantized row uses
 * round-to-nearest-even and clamps to [-127,127]. Each slab contains slabch
 * 16-byte table entries {data_off,0x00ffffff,scale,0}, then slabch*K row bytes.
 * data_off=slabch*16+channel*K is relative to the slab, relocated by the caller.
 * Total output size is N*16+N*K bytes. */
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
            const hpi_block_q8_0 *row = w + (size_t) n * (size_t) nb;

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

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t model_id;
    uint64_t tensor_id;
    uint64_t epoch;
    uint32_t type;
    uint32_t layout;
    uint32_t expert;
    uint32_t N;
    uint32_t K;
    uint32_t slab;
    uint32_t normalization;
} hpi3720_expert_key;

enum hpi3720_expert_state {
    HPI3720_EXPERT_EMPTY = 0,
    HPI3720_EXPERT_FILLING,
    HPI3720_EXPERT_READY,
};

typedef struct {
    hpi3720_expert_key key;
    enum hpi3720_expert_state state;
    uint64_t generation;
    uint32_t active_uses;
    uint64_t last_used;
} hpi3720_expert_slot;

typedef struct {
    hpi3720_expert_slot *slots;
    size_t slot_count;
    uint32_t records_per_slot;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t pending;
    uint64_t busy;
    uint64_t evictions;
    uint64_t fill_failures;
} hpi3720_expert_cache;

typedef struct {
    size_t slot;
    uint64_t generation;
} hpi3720_expert_ticket;

enum hpi3720_expert_action {
    HPI3720_EXPERT_HIT = 0,
    HPI3720_EXPERT_MISS_FILL,
    HPI3720_EXPERT_MISS_PENDING,
    HPI3720_EXPERT_MISS_BUSY,
};

static inline int hpi3720_expert_key_equal(const hpi3720_expert_key *a,
                                            const hpi3720_expert_key *b) {
    return a && b && a->model_id == b->model_id && a->tensor_id == b->tensor_id &&
           a->epoch == b->epoch && a->type == b->type && a->layout == b->layout &&
           a->expert == b->expert && a->N == b->N && a->K == b->K &&
           a->slab == b->slab && a->normalization == b->normalization;
}

static inline int hpi3720_expert_cache_init(hpi3720_expert_cache *cache,
                                             hpi3720_expert_slot *slots,
                                             size_t slot_count,
                                             uint32_t records_per_slot) {
    if (!cache || !slots || !slot_count || !records_per_slot ||
        slot_count > SIZE_MAX / sizeof *slots) return -1;
    memset(cache, 0, sizeof *cache);
    memset(slots, 0, slot_count * sizeof *slots);
    cache->slots = slots;
    cache->slot_count = slot_count;
    cache->records_per_slot = records_per_slot;
    return 0;
}

/* A HIT reserves one execution record until release_hit. MISS_FILL reserves an
 * idle allocation for the caller's asynchronous decoder. All miss actions mean
 * the current operation must stay on the optimized CPU path. */
static inline enum hpi3720_expert_action hpi3720_expert_request(
        hpi3720_expert_cache *cache, const hpi3720_expert_key *key,
        hpi3720_expert_ticket *ticket) {
    if (!cache || !cache->slots || !key || !ticket) return HPI3720_EXPERT_MISS_BUSY;
    ticket->slot = SIZE_MAX;
    ticket->generation = 0;
    for (size_t i = 0; i < cache->slot_count; ++i) {
        hpi3720_expert_slot *slot = &cache->slots[i];
        if (slot->state == HPI3720_EXPERT_EMPTY || !hpi3720_expert_key_equal(&slot->key, key)) continue;
        ticket->slot = i;
        ticket->generation = slot->generation;
        if (slot->state == HPI3720_EXPERT_FILLING) {
            ++cache->pending;
            return HPI3720_EXPERT_MISS_PENDING;
        }
        if (slot->active_uses >= cache->records_per_slot) {
            ++cache->busy;
            return HPI3720_EXPERT_MISS_BUSY;
        }
        ++slot->active_uses;
        slot->last_used = ++cache->clock;
        ++cache->hits;
        return HPI3720_EXPERT_HIT;
    }

    ++cache->misses;
    size_t victim = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < cache->slot_count; ++i) {
        hpi3720_expert_slot *slot = &cache->slots[i];
        if (slot->state == HPI3720_EXPERT_EMPTY) { victim = i; break; }
        if (slot->state == HPI3720_EXPERT_READY && !slot->active_uses && slot->last_used < oldest) {
            victim = i;
            oldest = slot->last_used;
        }
    }
    if (victim == SIZE_MAX) {
        ++cache->busy;
        return HPI3720_EXPERT_MISS_BUSY;
    }
    hpi3720_expert_slot *slot = &cache->slots[victim];
    if (slot->state == HPI3720_EXPERT_READY) ++cache->evictions;
    slot->key = *key;
    slot->state = HPI3720_EXPERT_FILLING;
    slot->active_uses = 0;
    slot->last_used = ++cache->clock;
    ++slot->generation;
    if (!slot->generation) ++slot->generation;
    ticket->slot = victim;
    ticket->generation = slot->generation;
    return HPI3720_EXPERT_MISS_FILL;
}

static inline int hpi3720_expert_fill_complete(hpi3720_expert_cache *cache,
                                                hpi3720_expert_ticket ticket,
                                                int success) {
    if (!cache || !cache->slots || ticket.slot >= cache->slot_count) return -1;
    hpi3720_expert_slot *slot = &cache->slots[ticket.slot];
    if (slot->state != HPI3720_EXPERT_FILLING || slot->generation != ticket.generation) return -1;
    if (success) {
        slot->state = HPI3720_EXPERT_READY;
        slot->last_used = ++cache->clock;
    } else {
        memset(&slot->key, 0, sizeof slot->key);
        slot->state = HPI3720_EXPERT_EMPTY;
        slot->last_used = 0;
        ++cache->fill_failures;
    }
    return 0;
}

static inline int hpi3720_expert_release_hit(hpi3720_expert_cache *cache,
                                              hpi3720_expert_ticket ticket) {
    if (!cache || !cache->slots || ticket.slot >= cache->slot_count) return -1;
    hpi3720_expert_slot *slot = &cache->slots[ticket.slot];
    if (slot->state != HPI3720_EXPERT_READY || slot->generation != ticket.generation ||
        !slot->active_uses) return -1;
    --slot->active_uses;
    return 0;
}

/* Opt in after npu.h declarations for the runtime or the pure SIMD oracle. */
#if defined(HPI_NPU_INTERNAL_SIMD)
/* Included after npu.h: its scalar conversions define rounding and NaN policy. */
#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>

static int hpi3720_has_f16c(void) {
    return __builtin_cpu_supports("avx") && __builtin_cpu_supports("f16c");
}

__attribute__((target("avx,f16c")))
static void hpi3720_pack_f16(uint16_t *dst, const float *src, size_t n) {
    size_t i = 0;
    for (; n - i >= 8; i += 8) {
        const __m256 x = _mm256_loadu_ps(src + i);
        const __m128i h = _mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i *)(void *)(dst + i), h);
        const unsigned nan = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
        if (nan) for (unsigned j = 0; j < 8; ++j)
            if (nan & (1u << j)) dst[i + j] = npu_f32_to_f16(src[i + j]);
    }
    for (; i < n; ++i) dst[i] = npu_f32_to_f16(src[i]);
}

__attribute__((target("avx,f16c")))
static void hpi3720_unpack_f16(float *dst, const uint16_t *src, size_t n) {
    size_t i = 0;
    for (; n - i >= 8; i += 8) {
        const __m128i h = _mm_loadu_si128((const __m128i *)(const void *)(src + i));
        const __m256 x = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, x);
        const unsigned nan = (unsigned)_mm256_movemask_ps(_mm256_cmp_ps(x, x, _CMP_UNORD_Q));
        // Preserve signaling/quiet payload bits exactly as the portable decoder.
        if (nan) for (unsigned j = 0; j < 8; ++j)
            if (nan & (1u << j)) dst[i + j] = npu_f16_to_f32(src[i + j]);
    }
    for (; i < n; ++i) dst[i] = npu_f16_to_f32(src[i]);
}
#else
static int hpi3720_has_f16c(void) { return 0; }
static void hpi3720_pack_f16(uint16_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = npu_f32_to_f16(src[i]);
}
static void hpi3720_unpack_f16(float *dst, const uint16_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = npu_f16_to_f32(src[i]);
}
#endif
#endif /* HPI_NPU_INTERNAL_SIMD */

#endif /* HPI_NPU_INTERNAL_H */
