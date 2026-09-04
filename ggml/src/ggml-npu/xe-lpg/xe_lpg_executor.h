#ifndef XE_LPG_EXECUTOR_H
#define XE_LPG_EXECUTOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xe_lpg_executor xe_lpg_executor;

typedef struct {
    uint64_t model_id;
    uint64_t epoch;
    uint64_t gate_tensor_id;
    uint64_t up_tensor_id;
    uint64_t down_tensor_id;
    uint32_t layer;
    uint32_t gate_type;
    uint32_t up_type;
    uint32_t down_type;
    uint32_t layout;
    uint32_t expert;
    uint32_t input_width;
    uint32_t hidden_width;
    uint32_t output_width;
} xe_lpg_expert_key;

typedef struct {
    xe_lpg_expert_key key;
    const void *gate;
    const void *up;
    const void *down;
} xe_lpg_expert;

typedef enum {
    XE_LPG_REPLAY_EXACT = 0,
    XE_LPG_REPLAY_WARMING,
    XE_LPG_REPLAY_UNAVAILABLE,
    XE_LPG_REPLAY_MISMATCH,
} xe_lpg_replay_status;

typedef struct {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t fill_bytes;
    uint64_t exact_replays;
    uint64_t mismatches;
    uint64_t failures;
} xe_lpg_profile;

xe_lpg_executor *xe_lpg_executor_create(const char *module_path, size_t cache_bytes);
void xe_lpg_executor_destroy(xe_lpg_executor *executor);
const char *xe_lpg_executor_error(const xe_lpg_executor *executor);
void xe_lpg_executor_profile(const xe_lpg_executor *executor, xe_lpg_profile *profile);

xe_lpg_replay_status xe_lpg_executor_replay(
        xe_lpg_executor *executor,
        const xe_lpg_expert experts[10],
        const void *input_q8_k,
        const float *expected_gate,
        const float *expected_up,
        const float expected_down[25600]);

#ifdef __cplusplus
}
#endif

#endif
