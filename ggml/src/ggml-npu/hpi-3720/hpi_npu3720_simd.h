#ifndef HPI_NPU3720_SIMD_H
#define HPI_NPU3720_SIMD_H

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
#endif
