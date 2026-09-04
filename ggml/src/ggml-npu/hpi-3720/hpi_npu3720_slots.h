/* Internal execution ownership for shape-only graphs. Requires npu.h declarations.
 * Registered weight allocations are immutable and owned by the weight registry.
 * A slot's X/Y/list cannot be reused until output conversion completes.
 */
#ifndef HPI_NPU3720_SLOTS_H
#define HPI_NPU3720_SLOTS_H

enum hpi3720_slot_state { HPI3720_IDLE, HPI3720_PREPARED, HPI3720_SUBMITTED };
typedef struct {
    void *x, *y;
    const void *bound_weights;
    ze_command_list_handle_t list;
    enum hpi3720_slot_state state;
} hpi3720_exec_slot;

typedef struct {
    npu_graph graph;
    npu_dev *device;
    uint8_t *blob;
    size_t blob_size;
    uint8_t digest[32];
    hpi3720_exec_slot slots[2];
    ze_command_list_handle_t init_list;
    int initialized;
} hpi3720_program;

static inline hpi3720_exec_slot *hpi3720_slot_acquire(hpi3720_program *program) {
    for (int i = 0; i < 2; ++i) {
        if (program->slots[i].state == HPI3720_IDLE) {
            program->slots[i].state = HPI3720_PREPARED;
            return &program->slots[i];
        }
    }
    return NULL;
}

static inline int hpi3720_slot_record(hpi3720_program *program, hpi3720_exec_slot *slot,
                                     void *weights, ze_command_queue_handle_t queue) {
    npu_graph *g = &program->graph;
    if (slot->state != HPI3720_PREPARED) return -1;
    if (slot->list && slot->bound_weights == weights && program->initialized) return 0;
    if (npu_list_destroy(g->d, &slot->list)) return -1;
    slot->bound_weights = NULL;
    /* An existing recorded list does not change when the graph is rebound.
     * Bind this slot and record a new execute command, as verified on 5540. */
    if (npu_graph_set_arg(g, 0, slot->x) || npu_graph_set_arg(g, 1, weights) ||
        npu_graph_set_arg(g, 2, slot->y)) return -1;
    if (!program->initialized) {
        if (npu_list_create(g->d, &program->init_list)) return -1;
        if (npu_list_graph_init(g, program->init_list)) return -1;
        if (npu_queue_run(g->d, queue, program->init_list, UINT64_MAX)) return -1;
        if (npu_list_destroy(g->d, &program->init_list)) return -1;
        program->initialized = 1;
    }
    if (npu_list_create(g->d, &slot->list) || npu_list_graph_exec(g, slot->list)) return -1;
    slot->bound_weights = weights;
    return 0;
}

/* Call only after the queue has completed. Lists/graphs release references first. */
static inline int hpi3720_program_destroy(hpi3720_program *program) {
    npu_dev *d = program->device;
    for (int i = 0; i < 2; ++i) {
        if (program->slots[i].state == HPI3720_SUBMITTED) return -1;
        if (program->slots[i].list && npu_list_destroy(d, &program->slots[i].list)) return -1;
    }
    if (program->init_list && npu_list_destroy(d, &program->init_list)) return -1;
    if (program->graph.h && npu_graph_destroy(&program->graph)) return -1;
    for (int i = 0; i < 2; ++i) {
        if (program->slots[i].x && npu_mem_free(d, program->slots[i].x)) return -1;
        program->slots[i].x = NULL;
        if (program->slots[i].y && npu_mem_free(d, program->slots[i].y)) return -1;
        program->slots[i].y = NULL;
    }
    free(program->blob);
    memset(program, 0, sizeof *program);
    return 0;
}

#endif
