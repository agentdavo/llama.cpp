/*
 * hpi_fp16.h — portable IEEE-754 half <-> float conversion for the HPI Q8_0 path.
 *
 * Header-only, no dependencies beyond <stdint.h>. Round-to-nearest-even on the f32->f16 path.
 * This is the software fallback; a hardware build may define HPI_FP16_HW to route through F16C
 * intrinsics (the src/npu.h direct-access toolkit uses _mm_cvtph_ps / _mm_cvtps_ph). Kept
 * self-contained on purpose so hpi-3720 builds on any C11 compiler with no CPU-feature flags.
 */
#ifndef HPI_FP16_H
#define HPI_FP16_H

#include <stdint.h>

/* uint16_t IEEE-754 binary16 -> float. Matches ggml_half semantics (ggml-common.h). */
static inline float hpi_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                         /* +/- zero */
        } else {
            /* subnormal half -> normalized float */
            uint32_t e = 0, m = mant;
            while (!(m & 0x400u)) { m <<= 1; e++; }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e + 1) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); /* inf / nan */
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    /* type-pun through memcpy, never an incompatible pointer cast (src/npu.h rule) */
    __builtin_memcpy(&f, &bits, sizeof f);
    return f;
}

/* float -> uint16_t IEEE-754 binary16, round-to-nearest-even, saturating to +/- inf. */
static inline uint16_t hpi_f32_to_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  exp  = (int32_t)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFFu) == 0xFFu) {                 /* inf / nan */
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    if (exp >= 0x1F) {                                  /* overflow -> inf */
        return (uint16_t)(sign | 0x7C00u);
    }
    if (exp <= 0) {                                     /* subnormal / underflow */
        if (exp < -10) return (uint16_t)sign;          /* too small -> +/- 0 */
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        const uint32_t round = 1u << (shift - 1);
        uint32_t v = mant + round;
        if ((mant & ((round << 1) - 1)) == round) v = mant + round - ((v >> shift) & 1u); /* ties-to-even */
        return (uint16_t)(sign | (v >> shift));
    }
    /* normal: round mantissa from 23 to 10 bits, ties-to-even */
    const uint32_t round = 0x1000u;                     /* half of the dropped 13-bit field */
    uint32_t v = mant + round + ((mant >> 13) & 1u);
    uint32_t out_exp  = (uint32_t)exp;
    uint32_t out_mant = v >> 13;
    if (out_mant & 0x400u) { out_mant = 0; out_exp++; } /* mantissa carry */
    if (out_exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (out_exp << 10) | (out_mant & 0x3FFu));
}

#endif /* HPI_FP16_H */
