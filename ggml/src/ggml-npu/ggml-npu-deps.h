#ifndef GGML_NPU_DEPS_H
#define GGML_NPU_DEPS_H

#include "ggml.h"
#include <stdint.h>

static inline bool ggml_npu_overlaps(const struct ggml_tensor *a, const struct ggml_tensor *b) {
    const size_t na = ggml_nbytes(a), nb = ggml_nbytes(b);
    if (!na || !nb) return false;
    if (!a->data || !b->data) return true;
    const uintptr_t pa = (uintptr_t)a->data, pb = (uintptr_t)b->data;
    // Subtraction avoids wrapping an end address; nbytes includes strided gaps.
    return pa <= pb ? pb - pa < na : pa - pb < nb;
}

static inline bool ggml_npu_metadata_op(enum ggml_op op) {
    return op == GGML_OP_NONE || op == GGML_OP_VIEW || op == GGML_OP_RESHAPE ||
           op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static inline bool ggml_npu_independent(const struct ggml_tensor *node, const struct ggml_tensor *dpu) {
    if (ggml_npu_metadata_op(node->op)) return true;
    // Only operators whose writes are confined to their destination are eligible.
    switch (node->op) {
        case GGML_OP_MUL_MAT: case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ADD: case GGML_OP_ADD1: case GGML_OP_SUB:
        case GGML_OP_MUL: case GGML_OP_DIV: case GGML_OP_SCALE:
        case GGML_OP_UNARY: case GGML_OP_GLU:
        case GGML_OP_NORM: case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX: case GGML_OP_ROPE:
        case GGML_OP_GET_ROWS: case GGML_OP_CONT: case GGML_OP_DUP:
            break;
        default: return false;
    }
    if (ggml_npu_overlaps(node, dpu)) return false;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        if (node->src[i] && ggml_npu_overlaps(node->src[i], dpu)) return false;
        if (dpu->src[i] && ggml_npu_overlaps(node, dpu->src[i])) return false;
    }
    return true;
}

#endif
