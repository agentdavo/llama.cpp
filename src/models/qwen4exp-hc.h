#pragma once

#include "ggml.h"

#include <cstdint>
#include <cstring>

#if defined(_MSC_VER)
#pragma float_control(precise, on, push)
#pragma fp_contract(off)
#endif

#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("fp-contract=off")))
#endif
static void qwen4exp_hc_write(
        ggml_tensor * dst, const ggml_tensor * residual, const ggml_tensor * block,
        const ggml_tensor * weights, int ith, int nth, void *) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const int64_t width = dst->ne[0];
    const int64_t branches = dst->ne[1];
    for (int64_t row = ith; row < branches * dst->ne[2]; row += nth) {
        const int64_t token = row / branches;
        const int64_t branch = row % branches;
        const char * r = static_cast<const char *>(residual->data) + branch * residual->nb[1] + token * residual->nb[2];
        const char * b = static_cast<const char *>(block->data) + token * block->nb[2];
        const char * w = static_cast<const char *>(weights->data) + branch * weights->nb[1] + token * weights->nb[2];
        char * out = static_cast<char *>(dst->data) + branch * dst->nb[1] + token * dst->nb[2];
        float weight;
        std::memcpy(&weight, w, sizeof(weight));
        for (int64_t i = 0; i < width; ++i) {
            float x, y;
            std::memcpy(&x, r + i * sizeof(float), sizeof(x));
            std::memcpy(&y, b + i * sizeof(float), sizeof(y));
            // Match separate MUL and ADD rounding, including when FMA is available.
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
            const float product = y * weight;
#else
            const volatile float product = y * weight;
#endif
            const float value = x + product;
            std::memcpy(out + i * sizeof(float), &value, sizeof(value));
        }
    }
}

#if defined(_MSC_VER)
#pragma float_control(pop)
#endif

static ggml_tensor * qwen4exp_hc_write_fused(
        ggml_context * ctx, ggml_tensor * residual, ggml_tensor * block, ggml_tensor * weights,
        bool inplace = false) {
    if (residual->type != GGML_TYPE_F32 || block->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 ||
        residual->nb[0] != sizeof(float) || block->nb[0] != sizeof(float) || weights->nb[0] != sizeof(float) ||
        residual->ne[3] != 1 || block->ne[3] != 1 || weights->ne[3] != 1 ||
        block->ne[0] != residual->ne[0] || block->ne[1] != 1 || block->ne[2] != residual->ne[2] ||
        weights->ne[0] != 1 || weights->ne[1] != residual->ne[1] || weights->ne[2] != residual->ne[2]) {
        return nullptr;
    }
    const int tasks = residual->ne[2] <= 4 ? 1 : GGML_N_TASKS_MAX;
    return inplace
        ? ggml_map_custom3_inplace(ctx, residual, block, weights, qwen4exp_hc_write, tasks, nullptr)
        : ggml_map_custom3(ctx, residual, block, weights, qwen4exp_hc_write, tasks, nullptr);
}
