#define ZE_GPU_IMPLEMENTATION
#include "ze_gpu.h"
#include "xe_lpg_executor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    XE_ROUTES = 10,
    XE_INPUT = 2560,
    XE_HIDDEN = 640,
    XE_OUTPUT = 2560,
    XE_GATE_BYTES = 921600,
    XE_UP_BYTES = 921600,
    XE_DOWN_BYTES = 1228800,
    XE_TRIPLET_BYTES = XE_GATE_BYTES + XE_UP_BYTES + XE_DOWN_BYTES,
};

typedef struct {
    xe_lpg_expert_key key;
    uint64_t age;
    int ready;
    int dirty;
} xe_lpg_slot;

struct xe_lpg_executor {
    ze_gpu gpu;
    ze_gpu_program gate;
    ze_gpu_program quant;
    ze_gpu_program down;
    ze_gpu_buffer buffers[10];
    xe_lpg_slot *slots;
    uint32_t capacity;
    uint64_t clock;
    xe_lpg_profile profile;
    char error[512];
};

static int key_equal(const xe_lpg_expert_key *a, const xe_lpg_expert_key *b) {
    return a->model_id == b->model_id && a->epoch == b->epoch &&
           a->gate_tensor_id == b->gate_tensor_id && a->up_tensor_id == b->up_tensor_id &&
           a->down_tensor_id == b->down_tensor_id && a->layer == b->layer &&
           a->gate_type == b->gate_type && a->up_type == b->up_type &&
           a->down_type == b->down_type && a->layout == b->layout &&
           a->expert == b->expert && a->input_width == b->input_width &&
           a->hidden_width == b->hidden_width && a->output_width == b->output_width;
}

static int read_module(const char *path, unsigned char **data, size_t *bytes) {
    FILE *file = fopen(path, "rb");
    long size = 0;
    *data = NULL;
    *bytes = 0;
    if (!file || fseek(file, 0, SEEK_END) || (size = ftell(file)) <= 0 ||
        size > 16 * 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        if (file) fclose(file);
        return -1;
    }
    *data = (unsigned char *)malloc((size_t)size);
    if (!*data || fread(*data, 1, (size_t)size, file) != (size_t)size || fclose(file)) {
        free(*data);
        *data = NULL;
        return -1;
    }
    *bytes = (size_t)size;
    return 0;
}

static int fail(xe_lpg_executor *executor, const char *what) {
    if (executor) {
        const char *detail = executor->gpu.error[0] ? executor->gpu.error : "no Level Zero detail";
        snprintf(executor->error, sizeof executor->error, "%.120s: %.360s", what, detail);
        executor->profile.failures++;
    }
    return -1;
}

static int find_slot(xe_lpg_executor *executor, const xe_lpg_expert_key *key, uint32_t *slot) {
    for (uint32_t i = 0; i < executor->capacity; ++i) {
        if (executor->slots[i].ready && key_equal(&executor->slots[i].key, key)) {
            *slot = i;
            return 1;
        }
    }
    return 0;
}

static uint32_t choose_slot(const xe_lpg_executor *executor, const uint8_t *protected_slots) {
    uint32_t victim = executor->capacity;
    for (uint32_t i = 0; i < executor->capacity; ++i) {
        if (protected_slots[i]) continue;
        if (!executor->slots[i].ready) return i;
        if (victim == executor->capacity || executor->slots[i].age < executor->slots[victim].age) victim = i;
    }
    return victim;
}

static int fill_slot(xe_lpg_executor *executor, uint32_t slot, const xe_lpg_expert *expert) {
    if (!expert->gate || !expert->up || !expert->down) return fail(executor, "null expert data");
    if (executor->slots[slot].ready) executor->profile.evictions++;
    memcpy((unsigned char *)executor->buffers[0].data + (size_t)slot * XE_GATE_BYTES, expert->gate, XE_GATE_BYTES);
    memcpy((unsigned char *)executor->buffers[1].data + (size_t)slot * XE_UP_BYTES, expert->up, XE_UP_BYTES);
    memcpy((unsigned char *)executor->buffers[2].data + (size_t)slot * XE_DOWN_BYTES, expert->down, XE_DOWN_BYTES);
    executor->slots[slot].key = expert->key;
    executor->slots[slot].age = ++executor->clock;
    executor->slots[slot].ready = 1;
    executor->slots[slot].dirty = 1;
    executor->profile.misses++;
    executor->profile.fill_bytes += XE_TRIPLET_BYTES;
    return 0;
}

xe_lpg_executor *xe_lpg_executor_create(const char *module_path, size_t cache_bytes) {
    xe_lpg_executor *executor = NULL;
    unsigned char *module = NULL;
    size_t module_bytes = 0;
    if (!module_path || cache_bytes < (size_t)XE_ROUTES * XE_TRIPLET_BYTES ||
        read_module(module_path, &module, &module_bytes)) return NULL;
    executor = (xe_lpg_executor *)calloc(1, sizeof *executor);
    if (!executor) goto error;
    size_t capacity = cache_bytes / XE_TRIPLET_BYTES;
    if (capacity > 512) capacity = 512;
    executor->capacity = (uint32_t)capacity;
    executor->slots = (xe_lpg_slot *)calloc(capacity, sizeof *executor->slots);
    if (!executor->slots || ze_gpu_open(&executor->gpu, 0) ||
        ze_gpu_program_create(&executor->gpu, ZE_MODULE_FORMAT_NATIVE, module, module_bytes, "gate_up", NULL, &executor->gate) ||
        ze_gpu_program_create(&executor->gpu, ZE_MODULE_FORMAT_NATIVE, module, module_bytes, "quantize_q8_1", NULL, &executor->quant) ||
        ze_gpu_program_create(&executor->gpu, ZE_MODULE_FORMAT_NATIVE, module, module_bytes, "down_q5", NULL, &executor->down)) goto error;
    const size_t sizes[10] = {
        capacity * XE_GATE_BYTES, capacity * XE_UP_BYTES, capacity * XE_DOWN_BYTES,
        2920, 40, (6400 + 32) * sizeof(float), (6400 + 32) * sizeof(float),
        (6400 + 32) * sizeof(float), 7200, (25600 + 32) * sizeof(float),
    };
    for (uint32_t i = 0; i < 10; ++i) if (ze_gpu_buffer_alloc(&executor->gpu, sizes[i], &executor->buffers[i])) goto error;
    free(module);
    return executor;
error:
    free(module);
    xe_lpg_executor_destroy(executor);
    return NULL;
}

void xe_lpg_executor_destroy(xe_lpg_executor *executor) {
    if (!executor) return;
    if (executor->gpu.pending) (void)ze_gpu_wait(&executor->gpu, 10000000000ull);
    for (uint32_t i = 0; i < 10; ++i) (void)ze_gpu_buffer_free(&executor->buffers[i]);
    (void)ze_gpu_program_destroy(&executor->down);
    (void)ze_gpu_program_destroy(&executor->quant);
    (void)ze_gpu_program_destroy(&executor->gate);
    (void)ze_gpu_close(&executor->gpu);
    free(executor->slots);
    free(executor);
}

const char *xe_lpg_executor_error(const xe_lpg_executor *executor) {
    return executor && executor->error[0] ? executor->error : "Xe-LPG executor unavailable";
}

void xe_lpg_executor_profile(const xe_lpg_executor *executor, xe_lpg_profile *profile) {
    if (profile) *profile = executor ? executor->profile : (xe_lpg_profile){0};
}

xe_lpg_replay_status xe_lpg_executor_replay(
        xe_lpg_executor *executor, const xe_lpg_expert experts[XE_ROUTES], const void *input_q8_k,
        const float *expected_gate, const float *expected_up, const float expected_down[25600]) {
    uint32_t slot_ids[XE_ROUTES];
    uint8_t *protected_slots = NULL;
    if (!executor || !experts || !input_q8_k || !expected_down) return XE_LPG_REPLAY_UNAVAILABLE;
    protected_slots = (uint8_t *)calloc(executor->capacity, 1);
    if (!protected_slots) { fail(executor, "cache protection allocation"); return XE_LPG_REPLAY_UNAVAILABLE; }
    for (uint32_t rank = 0; rank < XE_ROUTES; ++rank) {
        uint32_t slot = executor->capacity;
        if (find_slot(executor, &experts[rank].key, &slot)) {
            executor->profile.hits++;
            executor->slots[slot].age = ++executor->clock;
        } else {
            slot = choose_slot(executor, protected_slots);
            if (slot == executor->capacity || fill_slot(executor, slot, &experts[rank])) {
                free(protected_slots);
                return XE_LPG_REPLAY_UNAVAILABLE;
            }
        }
        protected_slots[slot] = 1;
        slot_ids[rank] = slot;
    }
    free(protected_slots);
    memcpy(executor->buffers[3].data, input_q8_k, 2920);
    memcpy(executor->buffers[4].data, slot_ids, sizeof slot_ids);
    const uint32_t one = 1;
    void *gate_args[5] = {executor->buffers[3].data, executor->buffers[4].data, executor->buffers[5].data,
                          executor->buffers[6].data, executor->buffers[7].data};
    if (ze_gpu_arg(&executor->gate, 0, sizeof(void *), &executor->buffers[0].data) ||
        ze_gpu_arg(&executor->gate, 1, sizeof(void *), &executor->buffers[1].data)) goto gpu_fail;
    for (uint32_t i = 0; i < 5; ++i) if (ze_gpu_arg(&executor->gate, i + 2, sizeof(void *), &gate_args[i])) goto gpu_fail;
    if (ze_gpu_arg(&executor->gate, 7, sizeof one, &one)) goto gpu_fail;
    uint32_t blocks = 200;
    if (ze_gpu_arg(&executor->quant, 0, sizeof(void *), &executor->buffers[7].data) ||
        ze_gpu_arg(&executor->quant, 1, sizeof(void *), &executor->buffers[8].data) ||
        ze_gpu_arg(&executor->quant, 2, sizeof blocks, &blocks) ||
        ze_gpu_arg(&executor->down, 0, sizeof(void *), &executor->buffers[2].data) ||
        ze_gpu_arg(&executor->down, 1, sizeof(void *), &executor->buffers[8].data) ||
        ze_gpu_arg(&executor->down, 2, sizeof(void *), &executor->buffers[4].data) ||
        ze_gpu_arg(&executor->down, 3, sizeof(void *), &executor->buffers[9].data) ||
        ze_gpu_arg(&executor->down, 4, sizeof one, &one)) goto gpu_fail;

    size_t range_sizes[2 + 3 * 512];
    const void *ranges[2 + 3 * 512];
    uint32_t range_count = 0;
    for (uint32_t slot = 0; slot < executor->capacity; ++slot) if (executor->slots[slot].dirty) {
        const size_t bytes[3] = {XE_GATE_BYTES, XE_UP_BYTES, XE_DOWN_BYTES};
        for (uint32_t role = 0; role < 3; ++role) {
            range_sizes[range_count] = bytes[role];
            ranges[range_count++] = (const unsigned char *)executor->buffers[role].data + (size_t)slot * bytes[role];
        }
        executor->slots[slot].dirty = 0;
    }
    range_sizes[range_count] = executor->buffers[3].size; ranges[range_count++] = executor->buffers[3].data;
    range_sizes[range_count] = executor->buffers[4].size; ranges[range_count++] = executor->buffers[4].data;
    const uint32_t local[3] = {8, 1, 1};
    const uint32_t gate_groups[3] = {6400, 1, 1};
    const uint32_t quant_groups[3] = {200, 1, 1};
    const uint32_t down_groups[3] = {25600, 1, 1};
    if (ze_gpu_record_memory_ranges(&executor->gpu, range_count, range_sizes, ranges) ||
        ze_gpu_record(&executor->gate, local, gate_groups) || ze_gpu_execute(&executor->gpu) ||
        ze_gpu_wait(&executor->gpu, 10000000000ull) || ze_gpu_record(&executor->quant, local, quant_groups)) goto gpu_fail;
    const size_t quant_bytes = executor->buffers[8].size;
    const void *quant_data = executor->buffers[8].data;
    if (ze_gpu_record_memory_ranges(&executor->gpu, 1, &quant_bytes, &quant_data) ||
        ze_gpu_record(&executor->down, local, down_groups) || ze_gpu_execute(&executor->gpu) ||
        ze_gpu_wait(&executor->gpu, 10000000000ull)) goto gpu_fail;
    if ((expected_gate && memcmp(executor->buffers[5].data, expected_gate, 6400 * sizeof(float))) ||
        (expected_up && memcmp(executor->buffers[6].data, expected_up, 6400 * sizeof(float))) ||
        memcmp(executor->buffers[9].data, expected_down, 25600 * sizeof(float))) {
        executor->profile.mismatches++;
        return XE_LPG_REPLAY_MISMATCH;
    }
    executor->profile.exact_replays++;
    return XE_LPG_REPLAY_EXACT;
gpu_fail:
    fail(executor, "Level Zero replay");
    return XE_LPG_REPLAY_UNAVAILABLE;
}
