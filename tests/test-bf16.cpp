#include "ggml-cpu.h"
#include "ggml.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static float float_from_bits(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint32_t float_to_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

int main() {
    static const uint16_t low_bits[] = {
        0x0000, 0x0001, 0x7ffe, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
    };

    std::vector<float> input;
    input.reserve(65536 * (sizeof(low_bits) / sizeof(low_bits[0])) + 3);
    input.push_back(0.0f); // exercise unaligned source and destination offsets
    for (uint32_t hi = 0; hi <= 0xffff; ++hi) {
        for (uint16_t lo : low_bits) {
            input.push_back(float_from_bits((hi << 16) | lo));
        }
    }
    input.push_back(float_from_bits(0x00000001));
    input.push_back(float_from_bits(0x807fffff));

    const int64_t n = (int64_t) input.size() - 1;
    std::vector<ggml_bf16_t> actual((size_t) n + 1);
    std::vector<ggml_bf16_t> expected((size_t) n + 1);
    ggml_cpu_fp32_to_bf16(input.data() + 1, actual.data() + 1, n);
    ggml_fp32_to_bf16_row_ref(input.data() + 1, expected.data() + 1, n);

    for (int64_t i = 0; i < n; ++i) {
        if (actual[(size_t) i + 1].bits != expected[(size_t) i + 1].bits) {
            std::fprintf(stderr,
                    "fp32 -> bf16 mismatch at %lld: fp32=0x%08x actual=0x%04x expected=0x%04x\n",
                    (long long) i, float_to_bits(input[(size_t) i + 1]),
                    actual[(size_t) i + 1].bits, expected[(size_t) i + 1].bits);
            return 1;
        }
    }

    std::vector<ggml_bf16_t> bf16(65537);
    std::vector<float> fp32(65537);
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        bf16[(size_t) bits + 1].bits = (uint16_t) bits;
    }
    ggml_cpu_bf16_to_fp32(bf16.data() + 1, fp32.data() + 1, 65536);
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const uint32_t actual_bits = float_to_bits(fp32[(size_t) bits + 1]);
        const uint32_t expected_bits = bits << 16;
        if (actual_bits != expected_bits) {
            std::fprintf(stderr,
                    "bf16 -> fp32 mismatch at 0x%04x: actual=0x%08x expected=0x%08x\n",
                    bits, actual_bits, expected_bits);
            return 1;
        }
    }

    return 0;
}
