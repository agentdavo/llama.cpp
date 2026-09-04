#include "ggml-npu-xe-lpg.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "xe-lpg/xe_lpg_executor.h"

#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct ggml_backend_npu_xe_lpg {
    xe_lpg_executor *executor;
    uint64_t model_id;
    uint64_t epoch;
    uint32_t block;
    bool disabled;
    uint64_t warming;
    uint64_t exact;
    uint64_t skipped;
    const ggml_tensor *gate;
    const ggml_tensor *up;
    const ggml_tensor *down;
    uint32_t rows;
    int32_t ids[4][10];
    unsigned char input_q8k[4][2920];
};

static bool env_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value && value[0] && value[0] != '0';
}

static bool parse_u32_env(const char *name, uint32_t fallback, uint32_t minimum, uint32_t maximum, uint32_t *value) {
    const char *text = std::getenv(name);
    if (!text || !text[0]) {
        *value = fallback;
        return true;
    }
    char *end = nullptr;
    errno = 0;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (errno || !end || *end || parsed < minimum || parsed > maximum) return false;
    *value = static_cast<uint32_t>(parsed);
    return true;
}

static int tensor_block(const ggml_tensor *tensor, const char *role, uint32_t *block) {
    if (!tensor || !role || !block) return 0;
    char expected[64];
    const int length = std::snprintf(expected, sizeof expected, "%s-%%u%%c", role);
    if (length <= 0 || static_cast<size_t>(length) >= sizeof expected) return 0;
    unsigned parsed = 0;
    char tail = 0;
    if (std::sscanf(tensor->name, expected, &parsed, &tail) != 1) return 0;
    *block = parsed;
    return 1;
}

static bool tensor_layout(const ggml_tensor *tensor, enum ggml_type type, int64_t k, int64_t n, int64_t experts) {
    return tensor && tensor->type == type && tensor->ne[0] == k && tensor->ne[1] == n && tensor->ne[2] == experts &&
           tensor->ne[3] == 1 && tensor->nb[0] == ggml_type_size(type) && tensor->nb[1] == ggml_row_size(type, k) &&
           tensor->nb[2] >= tensor->nb[1] * static_cast<size_t>(n);
}

ggml_backend_npu_xe_lpg *ggml_backend_npu_xe_lpg_create(const void *model_identity) {
    if (!env_enabled("GGML_NPU_XE_LPG_SHADOW")) return nullptr;
    const char *module = std::getenv("GGML_NPU_XE_LPG_MODULE");
    uint32_t cache_mib = 512;
    uint32_t block = 0;
    if (!module || !module[0] || !parse_u32_env("GGML_NPU_XE_LPG_CACHE_MIB", 512, 30, 4096, &cache_mib) ||
        !parse_u32_env("GGML_NPU_XE_LPG_SHADOW_BLOCK", 0, 0, 255, &block)) {
        std::fprintf(stderr, "xe-lpg: invalid shadow configuration\n");
        return nullptr;
    }
    ggml_backend_npu_xe_lpg *bridge = new ggml_backend_npu_xe_lpg {};
    bridge->model_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(model_identity));
    bridge->epoch = 1;
    bridge->block = block;
    bridge->executor = xe_lpg_executor_create(module, static_cast<size_t>(cache_mib) * 1024 * 1024);
    if (!bridge->executor) {
        std::fprintf(stderr, "xe-lpg: Level Zero shadow executor unavailable\n");
        delete bridge;
        return nullptr;
    }
    std::fprintf(stderr, "xe-lpg: shadow replay enabled for block %u with %u MiB packed cache; CPU output remains authoritative\n",
                 block, cache_mib);
    return bridge;
}

void ggml_backend_npu_xe_lpg_destroy(ggml_backend_npu_xe_lpg *bridge) {
    if (!bridge) return;
    xe_lpg_profile profile {};
    xe_lpg_executor_profile(bridge->executor, &profile);
    std::fprintf(stderr, "xe-lpg: shadow exact=%" PRIu64 " warming=%" PRIu64 " skipped=%" PRIu64
                 " hits=%" PRIu64 " misses=%" PRIu64 " evictions=%" PRIu64 " fill_bytes=%" PRIu64
                 " mismatches=%" PRIu64 " failures=%" PRIu64 "\n",
                 bridge->exact, bridge->warming, bridge->skipped, profile.hits, profile.misses,
                 profile.evictions, profile.fill_bytes, profile.mismatches, profile.failures);
    xe_lpg_executor_destroy(bridge->executor);
    delete bridge;
}

int ggml_backend_npu_xe_lpg_role(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node) {
    if (!bridge || bridge->disabled || !node || node->op != GGML_OP_MUL_MAT_ID) return 0;
    uint32_t block = 0;
    if (tensor_block(node, "ffn_moe_gate", &block) && block == bridge->block) return 1;
    if (tensor_block(node, "ffn_moe_up", &block) && block == bridge->block) return 2;
    if (tensor_block(node, "ffn_moe_down", &block) && block == bridge->block) return 3;
    return 0;
}

void ggml_backend_npu_xe_lpg_capture(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node, int role) {
    if (!bridge || bridge->disabled || !node || role < 1 || role > 3) return;
    if (role == 1) {
        bridge->gate = nullptr;
        bridge->up = nullptr;
        bridge->down = nullptr;
        const ggml_tensor *ids = node->src[2];
        const ggml_tensor *input = node->src[1];
        const ggml_type_traits_cpu *q8k = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K);
        if (!tensor_layout(node->src[0], GGML_TYPE_Q4_K, 2560, 640, 512) || !ids ||
            ids->type != GGML_TYPE_I32 || ids->ne[0] != 10 || ids->ne[1] < 1 || ids->ne[1] > 4 ||
            ids->nb[0] != sizeof(int32_t) ||
            !input || input->type != GGML_TYPE_F32 || input->ne[0] != 2560 || input->ne[1] != 1 ||
            input->ne[2] != ids->ne[1] || input->ne[3] != 1 || !ggml_is_contiguous(input) ||
            !q8k || !q8k->from_float ||
            ggml_row_size(GGML_TYPE_Q8_K, 2560) != sizeof bridge->input_q8k[0]) {
            bridge->skipped++;
            return;
        }
        bridge->rows = static_cast<uint32_t>(ids->ne[1]);
        for (uint32_t row = 0; row < bridge->rows; ++row) {
            q8k->from_float(reinterpret_cast<const float *>(static_cast<const char *>(input->data) + row * input->nb[2]),
                            bridge->input_q8k[row], 2560);
            for (uint32_t rank = 0; rank < 10; ++rank) {
                bridge->ids[row][rank] = *reinterpret_cast<const int32_t *>(
                        static_cast<const char *>(ids->data) + row * ids->nb[1] + rank * ids->nb[0]);
            }
        }
        bridge->gate = node;
    } else if (role == 2) {
        if (bridge->gate && tensor_layout(node->src[0], GGML_TYPE_Q4_K, 2560, 640, 512) &&
            node->src[1] == bridge->gate->src[1] && node->src[2] == bridge->gate->src[2]) bridge->up = node;
    } else {
        if (bridge->gate && bridge->up && tensor_layout(node->src[0], GGML_TYPE_Q5_1, 640, 2560, 512) &&
            node->src[2] == bridge->gate->src[2] && node->ne[0] == 2560 && node->ne[1] == 10 &&
            node->ne[2] == bridge->rows && node->ne[3] == 1 && ggml_is_contiguous(node) &&
            ggml_nbytes(node) == static_cast<size_t>(bridge->rows) * 25600 * sizeof(float)) bridge->down = node;
    }
}

void ggml_backend_npu_xe_lpg_complete(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down) {
    if (!bridge || bridge->disabled || !down || down != bridge->down || !bridge->gate || !bridge->up) return;
    const ggml_tensor *gate = bridge->gate;
    const ggml_tensor *up = bridge->up;
    for (uint32_t row = 0; row < bridge->rows; ++row) {
        xe_lpg_expert selected[10] {};
        for (uint32_t rank = 0; rank < 10; ++rank) {
            const int32_t expert = bridge->ids[row][rank];
            if (expert < 0 || expert >= 512) {
                bridge->disabled = true;
                std::fprintf(stderr, "xe-lpg: invalid expert id; shadow disabled\n");
                return;
            }
            selected[rank].key = {
                bridge->model_id, bridge->epoch,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gate->src[0]->data)),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(up->src[0]->data)),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(down->src[0]->data)),
                bridge->block, static_cast<uint32_t>(GGML_TYPE_Q4_K), static_cast<uint32_t>(GGML_TYPE_Q4_K),
                static_cast<uint32_t>(GGML_TYPE_Q5_1), 1, static_cast<uint32_t>(expert), 2560, 640, 2560,
            };
            selected[rank].gate = static_cast<const char *>(gate->src[0]->data) + static_cast<size_t>(expert) * gate->src[0]->nb[2];
            selected[rank].up = static_cast<const char *>(up->src[0]->data) + static_cast<size_t>(expert) * up->src[0]->nb[2];
            selected[rank].down = static_cast<const char *>(down->src[0]->data) + static_cast<size_t>(expert) * down->src[0]->nb[2];
        }
        const xe_lpg_replay_status status = xe_lpg_executor_replay(
                bridge->executor, selected, bridge->input_q8k[row], nullptr, nullptr,
                reinterpret_cast<const float *>(static_cast<const char *>(down->data) + row * down->nb[2]));
        if (status == XE_LPG_REPLAY_EXACT) {
            continue;
        }
        if (status == XE_LPG_REPLAY_WARMING) {
            bridge->warming++;
            continue;
        }
        bridge->disabled = true;
        std::fprintf(stderr, "xe-lpg: shadow replay failed at row %u status=%d (%s); disabled, CPU result retained\n",
                     row, static_cast<int>(status), xe_lpg_executor_error(bridge->executor));
        return;
    }
    bridge->exact++;
    bridge->gate = nullptr;
    bridge->up = nullptr;
    bridge->down = nullptr;
}
