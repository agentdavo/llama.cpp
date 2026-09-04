#include "hpi_npu3720_expert_cache.h"

#include <stdio.h>

static hpi3720_expert_key key(uint32_t expert) {
    hpi3720_expert_key k = { UINT64_C(0x1111), UINT64_C(0x2222), 7, 12, 1,
                            expert, 640, 2560, 32, 1024 };
    return k;
}

static int fill(hpi3720_expert_cache *cache, hpi3720_expert_key k) {
    hpi3720_expert_ticket ticket;
    return hpi3720_expert_request(cache, &k, &ticket) == HPI3720_EXPERT_MISS_FILL &&
           !hpi3720_expert_fill_complete(cache, ticket, 1);
}

int main(void) {
    hpi3720_expert_slot slots[2];
    hpi3720_expert_cache cache;
    if (hpi3720_expert_cache_init(&cache, slots, 2, 2)) return 1;

    hpi3720_expert_ticket ticket, stale;
    hpi3720_expert_key k0 = key(0), k1 = key(1), k2 = key(2);
    if (hpi3720_expert_request(&cache, &k0, &ticket) != HPI3720_EXPERT_MISS_FILL) return 1;
    stale = ticket;
    if (hpi3720_expert_request(&cache, &k0, &ticket) != HPI3720_EXPERT_MISS_PENDING) return 1;
    if (hpi3720_expert_fill_complete(&cache, stale, 1)) return 1;
    if (hpi3720_expert_request(&cache, &k0, &ticket) != HPI3720_EXPERT_HIT) return 1;
    hpi3720_expert_ticket second;
    if (hpi3720_expert_request(&cache, &k0, &second) != HPI3720_EXPERT_HIT) return 1;
    hpi3720_expert_ticket third;
    if (hpi3720_expert_request(&cache, &k0, &third) != HPI3720_EXPERT_MISS_BUSY) return 1;
    if (hpi3720_expert_release_hit(&cache, ticket) || hpi3720_expert_release_hit(&cache, second)) return 1;

    /* Fresh two-slot route: one hit, five off-path fills and three idle evictions. */
    if (hpi3720_expert_cache_init(&cache, slots, 2, 2)) return 1;
    const uint32_t route[] = { 0, 1, 0, 2, 1, 0 };
    for (size_t i = 0; i < sizeof route / sizeof route[0]; ++i) {
        const hpi3720_expert_key request = key(route[i]);
        const enum hpi3720_expert_action action = hpi3720_expert_request(&cache, &request, &ticket);
        if (action == HPI3720_EXPERT_HIT) {
            if (hpi3720_expert_release_hit(&cache, ticket)) return 1;
        } else if (action == HPI3720_EXPERT_MISS_FILL) {
            if (hpi3720_expert_fill_complete(&cache, ticket, 1)) return 1;
        } else return 1;
    }
    if (cache.hits != 1 || cache.misses != 5 || cache.evictions != 3) return 1;

    /* No eviction is legal while both resident slots have active executions. */
    if (hpi3720_expert_cache_init(&cache, slots, 2, 1) || !fill(&cache, k0) || !fill(&cache, k1)) return 1;
    hpi3720_expert_ticket use0, use1;
    if (hpi3720_expert_request(&cache, &k0, &use0) != HPI3720_EXPERT_HIT ||
        hpi3720_expert_request(&cache, &k1, &use1) != HPI3720_EXPERT_HIT ||
        hpi3720_expert_request(&cache, &k2, &ticket) != HPI3720_EXPERT_MISS_BUSY) return 1;
    if (hpi3720_expert_release_hit(&cache, use0) || hpi3720_expert_release_hit(&cache, use1)) return 1;

    /* Every identity field participates; an epoch change cannot reuse old decoded bytes. */
    if (hpi3720_expert_cache_init(&cache, slots, 1, 1) || !fill(&cache, k0)) return 1;
    hpi3720_expert_key changed = k0;
#define CHECK_FIELD(field, value) do { \
        changed = k0; changed.field = (value); \
        if (hpi3720_expert_request(&cache, &changed, &ticket) != HPI3720_EXPERT_MISS_FILL || \
            hpi3720_expert_fill_complete(&cache, ticket, 0)) return 1; \
        if (!fill(&cache, k0)) return 1; \
    } while (0)
    CHECK_FIELD(model_id, UINT64_C(0x3333));
    CHECK_FIELD(tensor_id, UINT64_C(0x4444));
    CHECK_FIELD(epoch, UINT64_C(8));
    CHECK_FIELD(type, UINT32_C(13));
    CHECK_FIELD(layout, UINT32_C(2));
    CHECK_FIELD(expert, UINT32_C(9));
    CHECK_FIELD(N, UINT32_C(672));
    CHECK_FIELD(K, UINT32_C(2816));
    CHECK_FIELD(slab, UINT32_C(16));
    CHECK_FIELD(normalization, UINT32_C(2048));
#undef CHECK_FIELD

    /* A stale worker completion cannot publish a reused slot. */
    if (hpi3720_expert_cache_init(&cache, slots, 1, 1)) return 1;
    if (hpi3720_expert_request(&cache, &k0, &stale) != HPI3720_EXPERT_MISS_FILL ||
        hpi3720_expert_fill_complete(&cache, stale, 0) ||
        hpi3720_expert_request(&cache, &k1, &ticket) != HPI3720_EXPERT_MISS_FILL ||
        !hpi3720_expert_fill_complete(&cache, stale, 1) ||
        hpi3720_expert_fill_complete(&cache, ticket, 1)) return 1;

    printf("PASS expert cache: route hits=1 misses=5 evictions=3; identity, busy, pending, stale safe\n");
    return 0;
}
