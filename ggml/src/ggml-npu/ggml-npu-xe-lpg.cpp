#include "ggml-npu-xe-lpg.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "xe-lpg/xe_lpg_executor.h"

#include <cerrno>
#include <chrono>
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
    bool replace_requested;
    bool replacement_ready;
    uint64_t warming;
    uint64_t exact;
    uint64_t skipped;
    uint64_t replaced;
    uint64_t replace_fallbacks;
    uint64_t full_prefills;
    uint64_t ready_gate_id;
    uint64_t ready_up_id;
    uint64_t ready_down_id;
    uint64_t ready_gate_stride;
    uint64_t ready_up_stride;
    uint64_t ready_down_stride;
    xe_lpg_profile armed_profile;
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
           tensor->nb[2] == tensor->nb[1] * static_cast<size_t>(n) && ggml_is_contiguous(tensor) &&
           ggml_nbytes(tensor) == tensor->nb[2] * static_cast<size_t>(experts);
}

static bool ids_live_extent(size_t rows, size_t row_stride, size_t nbytes) {
    const size_t row_bytes = 10 * sizeof(int32_t);
    if (rows < 1 || rows > 4) return false;
    if (rows == 1) return nbytes == row_bytes;
    if (row_stride < row_bytes || rows - 1 > (SIZE_MAX - row_bytes) / row_stride) return false;
    return nbytes == (rows - 1) * row_stride + row_bytes;
}

static const unsigned char exact_module_sha256[32] = {
    0xf7, 0xde, 0x15, 0xfe, 0x87, 0xe5, 0x34, 0x08, 0x9d, 0x39, 0x73, 0x54, 0x74, 0x40, 0x0b, 0x42,
    0x36, 0xa4, 0xef, 0xaf, 0xad, 0x4b, 0xc7, 0x55, 0x68, 0x3e, 0xd1, 0x21, 0xb5, 0xf2, 0xaf, 0x2a,
};

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
    bridge->replace_requested = env_enabled("GGML_NPU_XE_LPG_REPLACE");
    bridge->executor = xe_lpg_executor_create(
            module, static_cast<size_t>(cache_mib) * 1024 * 1024,
            bridge->replace_requested ? exact_module_sha256 : nullptr);
    if (!bridge->executor) {
        std::fprintf(stderr, "xe-lpg: Level Zero shadow executor unavailable\n");
        delete bridge;
        return nullptr;
    }
    if (bridge->replace_requested && xe_lpg_executor_capacity(bridge->executor) != 512) {
        std::fprintf(stderr, "xe-lpg: replacement requires all 512 experts resident; shadow remains CPU-authoritative\n");
        bridge->replace_requested = false;
    }
    std::fprintf(stderr, "xe-lpg: shadow replay enabled for block %u with %u MiB packed cache; replacement=%s\n",
                 block, cache_mib, bridge->replace_requested ? "requested but unarmed" : "off");
    return bridge;
}

void ggml_backend_npu_xe_lpg_destroy(ggml_backend_npu_xe_lpg *bridge) {
    if (!bridge) return;
    xe_lpg_profile profile {};
    xe_lpg_executor_profile(bridge->executor, &profile);
    const uint64_t timed_misses = bridge->full_prefills ? profile.misses - bridge->armed_profile.misses : 0;
    const uint64_t timed_evictions = bridge->full_prefills ? profile.evictions - bridge->armed_profile.evictions : 0;
    const uint64_t timed_fill_bytes = bridge->full_prefills ? profile.fill_bytes - bridge->armed_profile.fill_bytes : 0;
    std::fprintf(stderr, "xe-lpg: shadow exact=%" PRIu64 " warming=%" PRIu64 " skipped=%" PRIu64
                 " replaced=%" PRIu64 " replace_fallbacks=%" PRIu64 " full_prefills=%" PRIu64
                 " hits=%" PRIu64 " misses=%" PRIu64 " evictions=%" PRIu64 " fill_bytes=%" PRIu64
                 " timed_misses=%" PRIu64 " timed_evictions=%" PRIu64 " timed_fill_bytes=%" PRIu64
                 " mismatches=%" PRIu64 " failures=%" PRIu64 "\n",
                 bridge->exact, bridge->warming, bridge->skipped, bridge->replaced, bridge->replace_fallbacks,
                 bridge->full_prefills, profile.hits, profile.misses,
                 profile.evictions, profile.fill_bytes, timed_misses, timed_evictions, timed_fill_bytes,
                 profile.mismatches, profile.failures);
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
        const bool gate_layout = tensor_layout(node->src[0], GGML_TYPE_Q4_K, 2560, 640, 512);
        const bool ids_layout = ids && ids->type == GGML_TYPE_I32 && ids->ne[0] == 10 &&
                ids->ne[1] >= 1 && ids->ne[1] <= 4 && ids->ne[2] == 1 && ids->ne[3] == 1 &&
                ids->nb[0] == sizeof(int32_t) &&
                ids_live_extent(static_cast<size_t>(ids->ne[1]), ids->nb[1], ggml_nbytes(ids));
        const bool input_layout = input && ids && input->type == GGML_TYPE_F32 && input->ne[0] == 2560 &&
                input->ne[1] == 1 && input->ne[2] == ids->ne[1] && input->ne[3] == 1 &&
                ggml_is_contiguous(input) &&
                ggml_nbytes(input) == static_cast<size_t>(input->ne[2]) * 2560 * sizeof(float);
        const size_t q8k_row_bytes = ggml_row_size(GGML_TYPE_Q8_K, 2560);
        const bool q8k_layout = q8k && q8k->from_float && q8k_row_bytes == sizeof bridge->input_q8k[0];
        if (env_enabled("GGML_NPU_XE_LPG_CAPTURE_DIAGNOSTIC")) {
            std::fprintf(stderr, "xe-lpg: capture diagnostic gate_layout=%d ids_layout=%d input_layout=%d q8k_layout=%d\n",
                         static_cast<int>(gate_layout), static_cast<int>(ids_layout),
                         static_cast<int>(input_layout), static_cast<int>(q8k_layout));
            if (ids) {
                std::fprintf(stderr, "xe-lpg: ids type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                        "contiguous=%d nbytes=%zu dense_nbytes=%zu\n", static_cast<int>(ids->type),
                        static_cast<long long>(ids->ne[0]), static_cast<long long>(ids->ne[1]),
                        static_cast<long long>(ids->ne[2]), static_cast<long long>(ids->ne[3]),
                        ids->nb[0], ids->nb[1], ids->nb[2], ids->nb[3], static_cast<int>(ggml_is_contiguous(ids)),
                        ggml_nbytes(ids), static_cast<size_t>(ids->ne[1]) * 10 * sizeof(int32_t));
            }
            if (input) {
                std::fprintf(stderr, "xe-lpg: input type=%d ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] "
                        "contiguous=%d nbytes=%zu expected_nbytes=%zu\n", static_cast<int>(input->type),
                        static_cast<long long>(input->ne[0]), static_cast<long long>(input->ne[1]),
                        static_cast<long long>(input->ne[2]), static_cast<long long>(input->ne[3]),
                        input->nb[0], input->nb[1], input->nb[2], input->nb[3],
                        static_cast<int>(ggml_is_contiguous(input)), ggml_nbytes(input),
                        static_cast<size_t>(input->ne[2]) * 2560 * sizeof(float));
            }
            std::fprintf(stderr, "xe-lpg: q8k traits=%d from_float=%d row_bytes=%zu expected_row_bytes=%zu\n",
                         q8k != nullptr, q8k && q8k->from_float, q8k_row_bytes, sizeof bridge->input_q8k[0]);
            bridge->disabled = true;
            bridge->skipped++;
            return;
        }
        if (!gate_layout || !ids_layout || !input_layout || !q8k_layout) {
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

static bool make_expert(
        const ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *gate, const ggml_tensor *up,
        const ggml_tensor *down, uint32_t expert, xe_lpg_expert *selected) {
    if (!bridge || !gate || !up || !down || !selected || expert >= 512) return false;
    selected->key = {
        bridge->model_id, bridge->epoch,
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gate->src[0]->data)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(up->src[0]->data)),
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(down->src[0]->data)),
        gate->src[0]->nb[2], up->src[0]->nb[2], down->src[0]->nb[2],
        bridge->block, static_cast<uint32_t>(GGML_TYPE_Q4_K), static_cast<uint32_t>(GGML_TYPE_Q4_K),
        static_cast<uint32_t>(GGML_TYPE_Q5_1), 1, expert, 2560, 640, 2560,
    };
    selected->gate = static_cast<const char *>(gate->src[0]->data) + static_cast<size_t>(expert) * gate->src[0]->nb[2];
    selected->up = static_cast<const char *>(up->src[0]->data) + static_cast<size_t>(expert) * up->src[0]->nb[2];
    selected->down = static_cast<const char *>(down->src[0]->data) + static_cast<size_t>(expert) * down->src[0]->nb[2];
    return true;
}

static uint32_t graph_consumers(ggml_cgraph *graph, const ggml_tensor *tensor) {
    uint32_t count = 0;
    const int nodes = graph ? ggml_graph_n_nodes(graph) : 0;
    for (int i = 0; i < nodes; ++i) {
        const ggml_tensor *node = ggml_graph_node(graph, i);
        for (uint32_t src = 0; src < GGML_MAX_SRC; ++src) {
            if (node->src[src] == tensor) ++count;
        }
    }
    return count;
}

int ggml_backend_npu_xe_lpg_begin_replace(
        ggml_backend_npu_xe_lpg *bridge, ggml_cgraph *graph, int gate_index) {
    if (!bridge || bridge->disabled || !bridge->replace_requested || !bridge->replacement_ready) return 0;
    if (!graph || gate_index < 0 || gate_index + 3 >= ggml_graph_n_nodes(graph)) {
        bridge->replace_requested = false;
        bridge->replacement_ready = false;
        bridge->replace_fallbacks++;
        return -2;
    }
    ggml_tensor *gate = ggml_graph_node(graph, gate_index);
    ggml_tensor *up = ggml_graph_node(graph, gate_index + 1);
    ggml_tensor *middle = ggml_graph_node(graph, gate_index + 2);
    ggml_tensor *down = ggml_graph_node(graph, gate_index + 3);
    uint32_t middle_block = 0;
    if (!gate || !up || !middle || !down || !gate->src[0] || !gate->src[1] || !gate->src[2] ||
        !up->src[0] || !down->src[0] ||
        ggml_backend_npu_xe_lpg_role(bridge, gate) != 1 ||
        ggml_backend_npu_xe_lpg_role(bridge, up) != 2 ||
        ggml_backend_npu_xe_lpg_role(bridge, down) != 3 ||
        middle->op != GGML_OP_GLU || ggml_get_glu_op(middle) != GGML_GLU_OP_SWIGLU ||
        !tensor_block(middle, "ffn_moe_swiglu", &middle_block) ||
        middle_block != bridge->block || middle->src[0] != gate || middle->src[1] != up ||
        down->src[1] != middle || down->src[2] != gate->src[2] ||
        graph_consumers(graph, gate) != 1 || graph_consumers(graph, up) != 1 || graph_consumers(graph, middle) != 1) {
        bridge->replace_requested = false;
        bridge->replacement_ready = false;
        bridge->replace_fallbacks++;
        std::fprintf(stderr, "xe-lpg: replacement island validation failed; permanently disarmed\n");
        return -2;
    }
    if (gate->src[2]->ne[1] != 1) return -1;
    ggml_backend_npu_xe_lpg_capture(bridge, gate, 1);
    ggml_backend_npu_xe_lpg_capture(bridge, up, 2);
    ggml_backend_npu_xe_lpg_capture(bridge, down, 3);
    if (bridge->gate != gate || bridge->up != up || bridge->down != down || bridge->rows != 1 ||
        bridge->ready_gate_id != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gate->src[0]->data)) ||
        bridge->ready_up_id != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(up->src[0]->data)) ||
        bridge->ready_down_id != static_cast<uint64_t>(reinterpret_cast<uintptr_t>(down->src[0]->data)) ||
        bridge->ready_gate_stride != gate->src[0]->nb[2] || bridge->ready_up_stride != up->src[0]->nb[2] ||
        bridge->ready_down_stride != down->src[0]->nb[2]) {
        bridge->replace_requested = false;
        bridge->replacement_ready = false;
        bridge->replace_fallbacks++;
        std::fprintf(stderr, "xe-lpg: replacement tensor identity changed; permanently disarmed\n");
        return -2;
    }
    return 1;
}

int ggml_backend_npu_xe_lpg_replace(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down) {
    if (!bridge || bridge->disabled || !bridge->replacement_ready || !down || down != bridge->down ||
        !bridge->gate || !bridge->up || bridge->rows != 1) return 0;
    xe_lpg_expert selected[10] {};
    for (uint32_t rank = 0; rank < 10; ++rank) {
        const int32_t expert = bridge->ids[0][rank];
        if (expert < 0 || !make_expert(bridge, bridge->gate, bridge->up, down, static_cast<uint32_t>(expert), &selected[rank])) {
            bridge->replace_requested = false;
            bridge->replacement_ready = false;
            bridge->replace_fallbacks++;
            return 0;
        }
    }
    xe_lpg_profile before {};
    xe_lpg_executor_profile(bridge->executor, &before);
    const xe_lpg_replay_status status = xe_lpg_executor_replay(
            bridge->executor, selected, bridge->input_q8k[0], nullptr, nullptr, nullptr,
            static_cast<float *>(down->data), 0);
    xe_lpg_profile after {};
    xe_lpg_executor_profile(bridge->executor, &after);
    if (status == XE_LPG_REPLAY_EXACT && before.misses == after.misses && before.evictions == after.evictions &&
        before.fill_bytes == after.fill_bytes) {
        bridge->replaced++;
        ggml_backend_npu_xe_lpg_abort(bridge);
        return 1;
    }
    bridge->replace_requested = false;
    bridge->replacement_ready = false;
    bridge->replace_fallbacks++;
    if (status != XE_LPG_REPLAY_WARMING) bridge->disabled = true;
    std::fprintf(stderr, "xe-lpg: replacement failed status=%d (%s); permanently disarmed\n",
                 static_cast<int>(status), xe_lpg_executor_error(bridge->executor));
    return 0;
}

void ggml_backend_npu_xe_lpg_abort(ggml_backend_npu_xe_lpg *bridge) {
    if (!bridge) return;
    bridge->gate = nullptr;
    bridge->up = nullptr;
    bridge->down = nullptr;
    bridge->rows = 0;
}

void ggml_backend_npu_xe_lpg_complete(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down) {
    if (!bridge || bridge->disabled || !down || down != bridge->down || !bridge->gate || !bridge->up) return;
    const ggml_tensor *gate = bridge->gate;
    const ggml_tensor *up = bridge->up;
    bool prefilled = false;
    if (bridge->replace_requested && !bridge->replacement_ready) {
        xe_lpg_expert all[512] {};
        bool valid = true;
        for (uint32_t expert = 0; expert < 512; ++expert) valid = valid && make_expert(bridge, gate, up, down, expert, &all[expert]);
        const auto prefill_start = std::chrono::steady_clock::now();
        const int prefill_status = valid ? xe_lpg_executor_prefill(bridge->executor, all, 512) : -1;
        const double prefill_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - prefill_start).count();
        std::fprintf(stderr, "xe-lpg: full-cache prefill status=%d wall_ms=%.3f experts=512 bytes=1572864000\n",
                     prefill_status, prefill_ms);
        if (prefill_status) {
            bridge->replace_requested = false;
            std::fprintf(stderr, "xe-lpg: full-cache prefill failed; replacement permanently disarmed\n");
        } else {
            prefilled = true;
        }
    }
    for (uint32_t row = 0; row < bridge->rows; ++row) {
        xe_lpg_expert selected[10] {};
        for (uint32_t rank = 0; rank < 10; ++rank) {
            const int32_t expert = bridge->ids[row][rank];
            if (expert < 0 || !make_expert(bridge, gate, up, down, static_cast<uint32_t>(expert), &selected[rank])) {
                bridge->disabled = true;
                std::fprintf(stderr, "xe-lpg: invalid expert id; shadow disabled\n");
                return;
            }
        }
        if (prefilled && row == 0) {
            std::fprintf(stderr, "xe-lpg: exact canary selected_ids=");
            for (uint32_t rank = 0; rank < 10; ++rank) {
                std::fprintf(stderr, "%s%d", rank ? "," : "", bridge->ids[row][rank]);
            }
            std::fputc('\n', stderr);
        }
        const xe_lpg_replay_status status = xe_lpg_executor_replay(
                bridge->executor, selected, bridge->input_q8k[row], nullptr, nullptr,
                reinterpret_cast<const float *>(static_cast<const char *>(down->data) + row * down->nb[2]), nullptr, 1);
        if (status == XE_LPG_REPLAY_EXACT) continue;
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
    if (prefilled) {
        bridge->ready_gate_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(gate->src[0]->data));
        bridge->ready_up_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(up->src[0]->data));
        bridge->ready_down_id = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(down->src[0]->data));
        bridge->ready_gate_stride = gate->src[0]->nb[2];
        bridge->ready_up_stride = up->src[0]->nb[2];
        bridge->ready_down_stride = down->src[0]->nb[2];
        bridge->replacement_ready = true;
        bridge->full_prefills++;
        xe_lpg_executor_profile(bridge->executor, &bridge->armed_profile);
        std::fprintf(stderr, "xe-lpg: replacement armed after 512-expert prefill and exact CPU/GPU canary\n");
    }
    ggml_backend_npu_xe_lpg_abort(bridge);
}
