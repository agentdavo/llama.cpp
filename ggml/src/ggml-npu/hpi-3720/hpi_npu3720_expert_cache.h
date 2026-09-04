/* Bounded decoded-expert residency policy. Pure state only: callers own the
 * Level Zero allocations, command lists, queue, and synchronization. A miss
 * starts an off-path fill and must execute on the CPU for the current request.
 */
#ifndef HPI_NPU3720_EXPERT_CACHE_H
#define HPI_NPU3720_EXPERT_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint64_t model_id;
    uint64_t tensor_id;
    uint64_t epoch;
    uint32_t type;
    uint32_t layout;
    uint32_t expert;
    uint32_t N;
    uint32_t K;
    uint32_t slab;
    uint32_t normalization;
} hpi3720_expert_key;

enum hpi3720_expert_state {
    HPI3720_EXPERT_EMPTY = 0,
    HPI3720_EXPERT_FILLING,
    HPI3720_EXPERT_READY,
};

typedef struct {
    hpi3720_expert_key key;
    enum hpi3720_expert_state state;
    uint64_t generation;
    uint32_t active_uses;
    uint64_t last_used;
} hpi3720_expert_slot;

typedef struct {
    hpi3720_expert_slot *slots;
    size_t slot_count;
    uint32_t records_per_slot;
    uint64_t clock;
    uint64_t hits;
    uint64_t misses;
    uint64_t pending;
    uint64_t busy;
    uint64_t evictions;
    uint64_t fill_failures;
} hpi3720_expert_cache;

typedef struct {
    size_t slot;
    uint64_t generation;
} hpi3720_expert_ticket;

enum hpi3720_expert_action {
    HPI3720_EXPERT_HIT = 0,
    HPI3720_EXPERT_MISS_FILL,
    HPI3720_EXPERT_MISS_PENDING,
    HPI3720_EXPERT_MISS_BUSY,
};

static inline int hpi3720_expert_key_equal(const hpi3720_expert_key *a,
                                            const hpi3720_expert_key *b) {
    return a && b && a->model_id == b->model_id && a->tensor_id == b->tensor_id &&
           a->epoch == b->epoch && a->type == b->type && a->layout == b->layout &&
           a->expert == b->expert && a->N == b->N && a->K == b->K &&
           a->slab == b->slab && a->normalization == b->normalization;
}

static inline int hpi3720_expert_cache_init(hpi3720_expert_cache *cache,
                                             hpi3720_expert_slot *slots,
                                             size_t slot_count,
                                             uint32_t records_per_slot) {
    if (!cache || !slots || !slot_count || !records_per_slot ||
        slot_count > SIZE_MAX / sizeof *slots) return -1;
    memset(cache, 0, sizeof *cache);
    memset(slots, 0, slot_count * sizeof *slots);
    cache->slots = slots;
    cache->slot_count = slot_count;
    cache->records_per_slot = records_per_slot;
    return 0;
}

/* A HIT reserves one execution record until release_hit. MISS_FILL reserves an
 * idle allocation for the caller's asynchronous decoder. All miss actions mean
 * the current operation must stay on the optimized CPU path. */
static inline enum hpi3720_expert_action hpi3720_expert_request(
        hpi3720_expert_cache *cache, const hpi3720_expert_key *key,
        hpi3720_expert_ticket *ticket) {
    if (!cache || !cache->slots || !key || !ticket) return HPI3720_EXPERT_MISS_BUSY;
    ticket->slot = SIZE_MAX;
    ticket->generation = 0;
    for (size_t i = 0; i < cache->slot_count; ++i) {
        hpi3720_expert_slot *slot = &cache->slots[i];
        if (slot->state == HPI3720_EXPERT_EMPTY || !hpi3720_expert_key_equal(&slot->key, key)) continue;
        ticket->slot = i;
        ticket->generation = slot->generation;
        if (slot->state == HPI3720_EXPERT_FILLING) {
            ++cache->pending;
            return HPI3720_EXPERT_MISS_PENDING;
        }
        if (slot->active_uses >= cache->records_per_slot) {
            ++cache->busy;
            return HPI3720_EXPERT_MISS_BUSY;
        }
        ++slot->active_uses;
        slot->last_used = ++cache->clock;
        ++cache->hits;
        return HPI3720_EXPERT_HIT;
    }

    ++cache->misses;
    size_t victim = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (size_t i = 0; i < cache->slot_count; ++i) {
        hpi3720_expert_slot *slot = &cache->slots[i];
        if (slot->state == HPI3720_EXPERT_EMPTY) { victim = i; break; }
        if (slot->state == HPI3720_EXPERT_READY && !slot->active_uses && slot->last_used < oldest) {
            victim = i;
            oldest = slot->last_used;
        }
    }
    if (victim == SIZE_MAX) {
        ++cache->busy;
        return HPI3720_EXPERT_MISS_BUSY;
    }
    hpi3720_expert_slot *slot = &cache->slots[victim];
    if (slot->state == HPI3720_EXPERT_READY) ++cache->evictions;
    slot->key = *key;
    slot->state = HPI3720_EXPERT_FILLING;
    slot->active_uses = 0;
    slot->last_used = ++cache->clock;
    ++slot->generation;
    if (!slot->generation) ++slot->generation;
    ticket->slot = victim;
    ticket->generation = slot->generation;
    return HPI3720_EXPERT_MISS_FILL;
}

static inline int hpi3720_expert_fill_complete(hpi3720_expert_cache *cache,
                                                hpi3720_expert_ticket ticket,
                                                int success) {
    if (!cache || !cache->slots || ticket.slot >= cache->slot_count) return -1;
    hpi3720_expert_slot *slot = &cache->slots[ticket.slot];
    if (slot->state != HPI3720_EXPERT_FILLING || slot->generation != ticket.generation) return -1;
    if (success) {
        slot->state = HPI3720_EXPERT_READY;
        slot->last_used = ++cache->clock;
    } else {
        memset(&slot->key, 0, sizeof slot->key);
        slot->state = HPI3720_EXPERT_EMPTY;
        slot->last_used = 0;
        ++cache->fill_failures;
    }
    return 0;
}

static inline int hpi3720_expert_release_hit(hpi3720_expert_cache *cache,
                                              hpi3720_expert_ticket ticket) {
    if (!cache || !cache->slots || ticket.slot >= cache->slot_count) return -1;
    hpi3720_expert_slot *slot = &cache->slots[ticket.slot];
    if (slot->state != HPI3720_EXPERT_READY || slot->generation != ticket.generation ||
        !slot->active_uses) return -1;
    --slot->active_uses;
    return 0;
}

#endif /* HPI_NPU3720_EXPERT_CACHE_H */
