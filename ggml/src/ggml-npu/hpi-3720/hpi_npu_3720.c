/* Intel NPU 2.7 / VPU 3720 Q8_0 offload.
 * V2 caches own baked graphs. Opt-in v3 caches share a shape/layout program,
 * persistent per-weight X/Y/command records, and immutable registered weight
 * images. Disk cache identity is validated on cold lookup; resident weights are
 * bounded and idle least-recently-used entries may be evicted.
 * Missing entries are reported to ggml for optimized CPU execution.
 */
#include "hpi_backend.h"

#if defined(HPI_HAVE_NPU_3720)

/* ---------------------------------------------------------------------------------------------- *
 *  Bring in the proven direct-access toolkit. This TU owns the npu.h implementation (exactly one
 *  translation unit may, per src/npu.h). NPU_NO_GDN drops the AVX2 GDN host math we don't use here;
 *  the full Level Zero layer (sections 3-5) is required, so NPU_NO_ZE must NOT be set.
 * ---------------------------------------------------------------------------------------------- */
#define NPU_IMPLEMENTATION
#ifndef NPU_NO_GDN            /* the CMake HW build also defines this; guard to avoid a redefine warning */
#define NPU_NO_GDN
#endif
#include "npu.h"
#include "hpi_npu3720_simd.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "sha256/sha256.h"
#include "hpi_npu3720_slots.h"

/* Retain the existing conservative baked-graph cap. Shared programs have
 * separate program and immutable-weight registries; neither is a hardware limit. */
#ifndef NPU3720_CACHE_MAX
#define NPU3720_CACHE_MAX 200
#endif

#include "hpi_npu3720_blob.h"   /* hpi_npu3720_blob_io + (when a provider is compiled in) build_blob's prototype */

#if !defined(HPI_NPU3720_BLOB_READY)
/* No provider TU (hpi_npu3720_blob.c) compiled in: the seam is unavailable, so the backend reports
 * unavailable and HPI_BACKEND_AUTO falls back to the CPU reference. */
static hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                         uint8_t **blob, size_t *blob_len,
                                         hpi_npu3720_blob_io *io) {
    (void)op; (void)blob; (void)blob_len; (void)io;
    return HPI_UNAVAILABLE;   /* the one open piece — see the block comment above */
}
#endif

/* ---------------------------------------------------------------------------------------------- *
 *  Per-shape resident state: one compiled blob + its graph + its staged, device-visible buffers.
 * ---------------------------------------------------------------------------------------------- */
typedef struct {
    int         valid;
    const void *w_key;               /* W pointer this blob was built for (strategy A cache key) */
    int64_t     M, N, K;
    uint8_t    *blob;                /* owned */
    size_t      blob_len;
    npu_graph   g;
    hpi_npu3720_blob_io io;
    void       *x_mem;               /* device-visible input  buffer (bound to g.arg[io.arg_x]) */
    void       *y_mem;               /* device-visible output buffer (bound to g.arg[io.arg_y]) */
    int         initialized;         /* AppendGraphInitialize has run for this graph */
    ze_command_list_handle_t exec_list;  /* the AppendGraphExecute list — built once, re-submitted per call */
    int         exec_ready;          /* exec_list is built + closed (re-runnable) */
    ze_command_list_handle_t init_list;
    hpi3720_program *program;
    hpi3720_exec_slot records[2];   /* stable lists/buffers owned by this immutable weight */
    enum hpi3720_slot_state legacy_state;
    void *weight_mem;
    size_t weight_bytes;
    uint64_t last_used;
} npu3720_shape;

typedef struct {
    npu3720_shape *shape;
    hpi3720_exec_slot *record;      /* NULL for a legacy baked-weight graph */
    void *x_mem;
    void *y_mem;
    ze_command_list_handle_t exec_list;
} npu3720_prepared;

#define NPU3720_MISS_MAX 1024
#define NPU3720_WEIGHT_MAX 1024
#define NPU3720_PROGRAM_MAX 64
#define NPU3720_RETRY_PREPARE ((hpi_status)-5)
typedef struct {
    const void *w_key;
    int64_t capacity, N, K;
} npu3720_miss;

typedef struct {
    npu_ze                    z;
    npu_dev                   d;
    ze_command_queue_handle_t q;
    int                       q_ok;
    int                       failed; /* Stop further submission after a driver/ownership error. */
    npu3720_shape             cache[NPU3720_WEIGHT_MAX];
    int                       ncache;
    int                       nlegacy;
    hpi3720_program            programs[NPU3720_PROGRAM_MAX];
    int                       nprograms;
    npu3720_miss               misses[NPU3720_MISS_MAX];
    int                       nmisses;
    hpi_profile              *profile;
    size_t                    blob_bytes_resident;  /* sum of blob_len over graphs the device accepted;
                                                     * printed on a graph-create failure so the ceiling
                                                     * can be read as COUNT- or MEMORY-bound. */
    int simd;
    size_t weight_bytes_resident;
    size_t weight_budget_bytes;
    uint64_t cache_clock;
    uint64_t weight_evictions;
} npu3720_priv;

#define NPU3720_DEFAULT_WEIGHT_CACHE_MIB 512u

static int64_t shape_capacity(const hpi_q8_0_gemm *op) {
    return op->M == 1 ? 1 : (op->M <= 256 ? 256 : op->M);
}

/* ---------------------------------------------------------------------------------------------- *
 *  available() / open() / close()
 * ---------------------------------------------------------------------------------------------- */

/* Opt-in hardware bring-up probe. GGML_NPU_HW_PROBE=1 opens the REAL device (verifying we genuinely
 * talk to the Windows NPU driver) even before the silicon compute seam is built; gemm then honestly
 * delegates to the CPU reference (see npu_gemm). Off by default so normal runs keep the safe path. */
#if !defined(HPI_NPU3720_BLOB_READY)
static int npu_probe_open_requested(void) {
    const char *e = getenv("GGML_NPU_HW_PROBE");
    return e && e[0] && e[0] != '0';
}
#endif

/* Cheap capability report. MUST be 0 whenever a gemm cannot produce a correct result, because
 * HPI_BACKEND_AUTO selects this backend on a true here and the ggml layer asserts every gemm is
 * HPI_OK. So it is gated on the compute seam being real; only then does it matter whether a VPU is
 * physically present (probed at open). The probe opt-in is the one honest exception: it keeps gemm
 * correct by falling back to the CPU reference, so advertising the device does not break the assert. */
static int npu_available(void) {
#if defined(HPI_NPU3720_BLOB_READY)
    return 1;
#else
    return npu_probe_open_requested() ? 1 : 0;
#endif
}

static hpi_status npu_open(hpi_device *dev) {
    if (!dev) return HPI_EINVAL;
    npu3720_priv *p = (npu3720_priv *)calloc(1, sizeof *p);
    if (!p) return HPI_ENOMEM;
    p->profile = dev->profile.enabled ? &dev->profile : NULL;
    const char *simd = getenv("GGML_NPU_SIMD");
    p->simd = simd && simd[0] && simd[0] != '0' && hpi3720_has_f16c();
    unsigned long long cache_mib = NPU3720_DEFAULT_WEIGHT_CACHE_MIB;
    const char *cache_env = getenv("GGML_NPU_WEIGHT_CACHE_MIB");
    if (cache_env && cache_env[0]) {
        char *end = NULL;
        const unsigned long long parsed = strtoull(cache_env, &end, 10);
        if (end && !end[0] && parsed >= 1 && parsed <= 16384) cache_mib = parsed;
    }
    p->weight_budget_bytes = (size_t)cache_mib * 1024u * 1024u;

    /* Step 1: load the Level Zero loader and open the first VPU device (+ context + graph table). */
    if (npu_ze_load(&p->z, 0) != 0) { free(p); return HPI_EDEVICE; }
    if (npu_dev_open(&p->z, &p->d) != 0) { free(p); return HPI_EDEVICE; }
    if (!p->d.graph) {                      /* no ZE_GRAPH_EXT table -> cannot run compiled blobs */
        free(p);
        return HPI_EDEVICE;
    }

    /* One non-turbo queue, with a host wait after each batch. Non-turbo on purpose: the turbo
     * clock + a metric streamer once hard-froze the machine (src/npu.h HAZARD); leave turbo to an
     * explicit, coordinated benchmark, not the default compute path. */
    if (npu_queue_create(&p->d, 0, 0, &p->q) != 0) { free(p); return HPI_EDEVICE; }
    p->q_ok = 1;

    /* Report the driver-reported identity once: this name/id can only come from the real Windows NPU
     * driver via Level Zero — proof the open path is genuine, not a fake. */
    fprintf(stderr, "hpi-3720: opened Intel NPU via Level Zero — \"%s\" id=0x%04x%s\n",
            p->d.props.name, (unsigned)p->d.props.deviceId, p->d.graph ? " (graph-ext)" : "");

    dev->priv = p;
    return HPI_OK;
}

/* Only release allocations after every command/graph reference is gone. On a
 * driver cleanup failure retain the remaining ownership record for process exit. */
static int shape_destroy(npu3720_priv *p, npu3720_shape *s) {
    if (s->program) {
        if (hpi3720_slots_destroy(s->program, s->records, 2)) return -1;
    } else {
        if (npu_list_destroy(&p->d, &s->exec_list) || npu_list_destroy(&p->d, &s->init_list)) return -1;
        if (npu_graph_destroy(&s->g)) return -1;
        if (s->x_mem && npu_mem_free(&p->d, s->x_mem)) return -1;
        s->x_mem = NULL;
        if (s->y_mem && npu_mem_free(&p->d, s->y_mem)) return -1;
        s->y_mem = NULL;
    }
    if (s->weight_mem && npu_mem_free(&p->d, s->weight_mem)) return -1;
    s->weight_mem = NULL;
    free(s->io.weight_image); s->io.weight_image = NULL;
    free(s->blob); s->blob = NULL;
    return 0;
}

static int shape_idle(const npu3720_shape *s) {
    if (!s->valid) return 0;
    if (!s->program) return s->legacy_state == HPI3720_IDLE;
    return s->records[0].state == HPI3720_IDLE && s->records[1].state == HPI3720_IDLE;
}

/* Make room only from completed runtime-weight records. Fixed array slots keep
 * prepared pointers stable while a batch is being assembled. Returns 1 when
 * every evictable entry is currently in flight, so the caller can drain once. */
static int weight_cache_make_room(npu3720_priv *p, size_t bytes) {
    if (bytes > p->weight_budget_bytes) return -2;
    while (p->weight_bytes_resident > p->weight_budget_bytes - bytes) {
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < NPU3720_WEIGHT_MAX; ++i) {
            const npu3720_shape *s = &p->cache[i];
            if (s->program && shape_idle(s) && s->last_used < oldest) {
                victim = i;
                oldest = s->last_used;
            }
        }
        if (victim < 0) return 1;
        npu3720_shape *s = &p->cache[victim];
        const size_t released = s->weight_bytes;
        if (shape_destroy(p, s)) { p->failed = 1; return -1; }
        memset(s, 0, sizeof *s);
        p->weight_bytes_resident -= released;
        --p->ncache;
        ++p->weight_evictions;
    }
    return 0;
}

static void npu_close(hpi_device *dev) {
    if (!dev || !dev->priv) return;
    npu3720_priv *p = (npu3720_priv *)dev->priv;
    if (p->q && p->z.zeCommandQueueSynchronize(p->q, UINT64_MAX) != ZE_RESULT_SUCCESS) goto retain;
    for (int i = 0; i < NPU3720_WEIGHT_MAX; ++i) if (p->cache[i].valid) {
        p->cache[i].legacy_state = HPI3720_IDLE;
        for (int j = 0; j < 2; ++j) p->cache[i].records[j].state = HPI3720_IDLE;
    }
    for (int i = 0; i < NPU3720_WEIGHT_MAX; ++i)
        if (p->cache[i].valid && shape_destroy(p, &p->cache[i])) goto retain;
    for (int i = 0; i < p->nprograms; ++i)
        if (hpi3720_program_destroy(&p->programs[i])) goto retain;
    if (npu_queue_destroy(&p->d, &p->q) || npu_dev_close(&p->d) || npu_ze_unload(&p->z)) goto retain;
    free(p); dev->priv = NULL;
    return;
retain:
    p->failed = 1;
    fprintf(stderr, "hpi-3720: cleanup failed; retaining resources still owned by the driver\n");
}

/* ---------------------------------------------------------------------------------------------- *
 *  Argument staging helpers (pure conversion between host f32 and the blob's device precision).
 * ---------------------------------------------------------------------------------------------- */

/* Copy `n` host floats into device buffer `dst` in the graph-argument precision `prec`. */
static hpi_status stage_in_f32(void *dst, ze_graph_argument_precision_t prec,
                               const float *src, size_t n, int simd) {
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
        memcpy(dst, src, n * sizeof(float));
        return HPI_OK;
    }
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
        uint16_t *d16 = (uint16_t *)dst;
        if (simd) hpi3720_pack_f16(d16, src, n);
        else for (size_t i = 0; i < n; i++) d16[i] = npu_f32_to_f16(src[i]);
        return HPI_OK;
    }
    return HPI_EDEVICE;   /* blob wants a precision the plumbing doesn't convert to yet */
}

/* Copy `n` values out of device buffer `src` (precision `prec`) into host floats `dst`. */
static hpi_status stage_out_f32(float *dst, ze_graph_argument_precision_t prec,
                                const void *src, size_t n, int simd) {
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP32) {
        memcpy(dst, src, n * sizeof(float));
        return HPI_OK;
    }
    if (prec == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
        const uint16_t *s16 = (const uint16_t *)src;
        if (simd) hpi3720_unpack_f16(dst, s16, n);
        else for (size_t i = 0; i < n; i++) dst[i] = npu_f16_to_f32(s16[i]);
        return HPI_OK;
    }
    return HPI_EDEVICE;
}

/* Build a compiled blob + graph + staged buffers for this shape and register it in the cache.
 * Returns the new entry, or NULL on failure (with *st set). */
static hpi3720_program *program_get(npu3720_priv *p, uint8_t **blob, size_t bytes,
    size_t weight_bytes, const hpi_q8_0_gemm *op) {
    uint8_t digest[32];
    sha256_hash(digest, *blob, bytes);
    for (int i = 0; i < p->nprograms; ++i) {
        hpi3720_program *program = &p->programs[i];
        if (!memcmp(program->digest, digest, 32)) {
            free(*blob); *blob = NULL;
            return program;
        }
    }
    if (p->nprograms == NPU3720_PROGRAM_MAX) return NULL;
    hpi3720_program *program = &p->programs[p->nprograms];
    ++p->nprograms;
    program->device = &p->d;
    program->blob = *blob; *blob = NULL;
    program->blob_size = bytes;
    if (npu_graph_create(&p->d, program->blob, bytes, &program->graph)) goto failed;
    const npu_graph *g = &program->graph;
    if (g->nargs != 3 || g->arg[0].is_output || g->arg[1].is_output || !g->arg[2].is_output ||
        g->arg[0].precision != ZE_GRAPH_ARGUMENT_PRECISION_FP16 ||
        g->arg[1].precision != ZE_GRAPH_ARGUMENT_PRECISION_UINT8 ||
        g->arg[2].precision != ZE_GRAPH_ARGUMENT_PRECISION_FP16 || op->M != 1 ||
        g->arg[0].elems != (size_t)op->K || g->arg[1].bytes != weight_bytes ||
        g->arg[2].elems != (size_t)op->N) {
        goto failed;
    }
    memcpy(program->digest, digest, 32);
    p->blob_bytes_resident += bytes;
    return program;
failed:
    if (hpi3720_program_destroy(program)) p->failed = 1;
    else --p->nprograms;
    return NULL;
}

static npu3720_shape *shape_build(npu3720_priv *p, const hpi_q8_0_gemm *op, hpi_status *st) {
    if (p->ncache >= NPU3720_WEIGHT_MAX) { *st = HPI_UNAVAILABLE; return NULL; }
    int cache_index = -1;
    for (int i = 0; i < NPU3720_WEIGHT_MAX; ++i) if (!p->cache[i].valid) { cache_index = i; break; }
    if (cache_index < 0) { *st = HPI_UNAVAILABLE; return NULL; }
    npu3720_shape tmp;
    memset(&tmp, 0, sizeof tmp);
    tmp.w_key = op->w; tmp.M = shape_capacity(op); tmp.N = op->N; tmp.K = op->K;
    *st = hpi_npu3720_build_blob(op, &tmp.blob, &tmp.blob_len, &tmp.io);
    if (*st != HPI_OK) return NULL;
    if (tmp.io.arg_w >= 0) {
        if (!tmp.io.weight_image || !tmp.io.weight_image_bytes ||
            tmp.io.arg_x != 0 || tmp.io.arg_w != 1 || tmp.io.arg_y != 2) goto unavailable;
        const int room = weight_cache_make_room(p, tmp.io.weight_image_bytes);
        if (room != 0) {
            *st = room == 1 ? NPU3720_RETRY_PREPARE : (room == -2 ? HPI_UNAVAILABLE : HPI_EDEVICE);
            goto cleanup;
        }
        tmp.program = program_get(p, &tmp.blob, tmp.blob_len, tmp.io.weight_image_bytes, op);
        if (!tmp.program) goto unavailable;
        tmp.weight_mem = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, tmp.io.weight_image_bytes);
        if (!tmp.weight_mem) goto unavailable;
        memcpy(tmp.weight_mem, tmp.io.weight_image, tmp.io.weight_image_bytes);
        free(tmp.io.weight_image); tmp.io.weight_image = NULL;
        for (int i = 0; i < 2; ++i) {
            tmp.records[i].x = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, tmp.program->graph.arg[0].bytes);
            tmp.records[i].y = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, tmp.program->graph.arg[2].bytes);
            if (!tmp.records[i].x || !tmp.records[i].y) goto unavailable;
        }
        tmp.g = tmp.program->graph; /* metadata view; the shared program owns the graph handle */
        tmp.blob_len = 0;
        tmp.weight_bytes = tmp.io.weight_image_bytes;
    } else {
        if (p->nlegacy >= NPU3720_CACHE_MAX) goto unavailable;
        if (npu_graph_create(&p->d, tmp.blob, tmp.blob_len, &tmp.g)) goto unavailable;
        if (tmp.io.arg_x < 0 || tmp.io.arg_y < 0 ||
            (uint32_t)tmp.io.arg_x >= tmp.g.nargs || (uint32_t)tmp.io.arg_y >= tmp.g.nargs ||
            tmp.g.arg[tmp.io.arg_x].is_output || !tmp.g.arg[tmp.io.arg_y].is_output) goto unavailable;
        const npu_graph_arg *ax = &tmp.g.arg[tmp.io.arg_x];
        const npu_graph_arg *ay = &tmp.g.arg[tmp.io.arg_y];
        if (ax->elems % (size_t)op->K || ax->elems / (size_t)op->K != (size_t)tmp.M ||
            ay->elems != (size_t)tmp.M * (size_t)op->N) goto unavailable;
        tmp.x_mem = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, ax->bytes);
        tmp.y_mem = npu_mem_alloc(&p->d, NPU_MEM_HOST, 0, ay->bytes);
        if (!tmp.x_mem || !tmp.y_mem) goto unavailable;
        if (npu_graph_set_arg(&tmp.g, (uint32_t)tmp.io.arg_x, tmp.x_mem) ||
            npu_graph_set_arg(&tmp.g, (uint32_t)tmp.io.arg_y, tmp.y_mem)) goto unavailable;
        ++p->nlegacy;
        p->blob_bytes_resident += tmp.blob_len;
    }
    tmp.valid = 1;
    tmp.last_used = ++p->cache_clock;
    p->cache[cache_index] = tmp;
    ++p->ncache;
    p->weight_bytes_resident += tmp.weight_bytes;
    *st = HPI_OK;
    return &p->cache[cache_index];

unavailable:
    *st = HPI_UNAVAILABLE;
cleanup:
    if (shape_destroy(p, &tmp)) {
        tmp.valid = 1;
        p->cache[cache_index] = tmp;
        ++p->ncache;
        p->failed = 1;
    }
    if (p->failed) *st = HPI_EDEVICE;
    return NULL;
}

static npu3720_shape *shape_find(npu3720_priv *p, const hpi_q8_0_gemm *op) {
    for (int i = 0; i < NPU3720_WEIGHT_MAX; i++) {
        npu3720_shape *s = &p->cache[i];
        if (s->valid && s->w_key == op->w && s->M == shape_capacity(op) && s->N == op->N && s->K == op->K) {
            s->last_used = ++p->cache_clock;
            return s;
        }
    }
    return NULL;
}

/* ---------------------------------------------------------------------------------------------- *
 *  gemm() — one validated Q8_0 GEMM (shapes already checked by the dispatcher).
 * ---------------------------------------------------------------------------------------------- */
/* Prepare a validated op: resolve/build its shape, stage X, ensure the graph is initialized and its
 * (re-runnable) exec list built. Returns the ready shape — its exec_list can then be submitted — or
 * NULL with *st set (HPI_UNAVAILABLE = no DPU blob, caller runs it on the CPU reference; else a real
 * device error). Does NOT submit or read back, so many ops can be prepared then submitted as a batch. */
static npu3720_shape *npu_prepare(npu3720_priv *p, const hpi_q8_0_gemm *op,
                                  hpi_status *st, npu3720_prepared *prepared) {
    *st = HPI_OK;
    memset(prepared, 0, sizeof *prepared);
    double t0 = p->profile ? npu_now_ms() : 0.0;
    int lookup_charged = 0;
    if (p->failed) { *st = HPI_EDEVICE; return NULL; }
    npu3720_shape *s = shape_find(p, op);
    if (s && p->profile) p->profile->warm_hits++;
    if (!s) {
        if (p->profile) p->profile->cache_misses++;
        /* Fixed weights/cache for this device lifetime. Avoid rehashing an entire
         * unavailable tensor on every token. Restart after changing the cache.
         * A full negative cache conservatively declines new cold shapes. */
        const int64_t capacity = shape_capacity(op);
        int missing = p->nmisses == NPU3720_MISS_MAX;
        for (int i = 0; !missing && i < p->nmisses; ++i) {
            const npu3720_miss *m = &p->misses[i];
            missing = m->w_key == op->w && m->capacity == capacity && m->N == op->N && m->K == op->K;
        }
        if (missing) {
            if (p->profile) p->profile->negative_hits++;
            *st = HPI_UNAVAILABLE;
        } else {
            if (p->profile) {
                p->profile->lookup_ms += npu_now_ms() - t0;
                p->profile->shape_builds++;
                lookup_charged = 1;
            }
            const double build_start = p->profile ? npu_now_ms() : 0.0;
            s = shape_build(p, op, st);
            if (p->profile) p->profile->build_ms += npu_now_ms() - build_start;
            if (!s && *st == HPI_UNAVAILABLE) {
                npu3720_miss *m = &p->misses[p->nmisses++];
                m->w_key = op->w; m->capacity = capacity; m->N = op->N; m->K = op->K;
            }
        }
    }
    if (p->profile) {
        const double elapsed = npu_now_ms() - t0;
        p->profile->prepare_ms += elapsed;
        /* A cold attempt charged lookup before shape_build above. */
        if (!lookup_charged) p->profile->lookup_ms += elapsed;
    }
    if (!s) return NULL;

    prepared->shape = s;
    if (s->program) {
        prepared->record = hpi3720_slot_acquire(s->records, 2);
        if (!prepared->record) { *st = NPU3720_RETRY_PREPARE; return NULL; }
        prepared->x_mem = prepared->record->x;
        prepared->y_mem = prepared->record->y;
    } else {
        if (s->legacy_state != HPI3720_IDLE) { *st = NPU3720_RETRY_PREPARE; return NULL; }
        s->legacy_state = HPI3720_PREPARED;
        prepared->x_mem = s->x_mem;
        prepared->y_mem = s->y_mem;
    }
    const npu_graph_arg *ax = &s->g.arg[s->io.arg_x];
    /* Stage X into the device input buffer in the blob's precision. When this op uses fewer rows than
     * the shared blob's M (an M=256 prefill blob serving op->M<256), zero the buffer first so the
     * unused rows compute on zeros — their output rows are simply not read back. */
    const size_t staged = (size_t)op->M * (size_t)op->K;
    t0 = p->profile ? npu_now_ms() : 0.0;
    if (ax->elems > staged) memset(prepared->x_mem, 0, ax->bytes);
    hpi_status cst = stage_in_f32(prepared->x_mem, ax->precision, op->x, staged, p->simd);
    if (p->profile) p->profile->input_ms += npu_now_ms() - t0;
    if (cst != HPI_OK) {
        if (prepared->record) prepared->record->state = HPI3720_IDLE;
        else s->legacy_state = HPI3720_IDLE;
        memset(prepared, 0, sizeof *prepared);
        *st = cst; return NULL;
    }
    if (s->program) {
        t0 = p->profile ? npu_now_ms() : 0.0;
        const int reuse = prepared->record->list && prepared->record->bound_weights == s->weight_mem && s->program->initialized;
        double init_elapsed = 0.0;
        const int record_status = hpi3720_slot_record(s->program, prepared->record, s->weight_mem, p->q,
                                                    p->profile ? &init_elapsed : NULL);
        if (p->profile) {
            const double elapsed = npu_now_ms() - t0;
            p->profile->prepare_ms += elapsed;
            p->profile->init_ms += init_elapsed;
            p->profile->record_ms += elapsed - init_elapsed;
            if (reuse) p->profile->list_reuses++;
            else if (!record_status) p->profile->command_records++;
        }
        if (record_status) {
            prepared->record->state = HPI3720_IDLE;
            memset(prepared, 0, sizeof *prepared);
            p->failed = 1;
            *st = HPI_EDEVICE; return NULL;
        }
        prepared->exec_list = prepared->record->list;
        return s;
    }

    t0 = p->profile ? npu_now_ms() : 0.0;
    /* AppendGraphInitialize once per shape (loads baked weights / builds descriptors). */
    if (!s->initialized) {
        if (npu_list_create(&p->d, &s->init_list) || npu_list_graph_init(&s->g, s->init_list) ||
            npu_queue_run(&p->d, p->q, s->init_list, UINT64_MAX) ||
            npu_list_destroy(&p->d, &s->init_list)) { p->failed = 1; *st = HPI_EDEVICE; return NULL; }
        s->initialized = 1;
        if (p->profile) {
            const double now = npu_now_ms();
            p->profile->init_ms += now - t0;
            p->profile->prepare_ms += now - t0;
            t0 = now;
        }
    }
    /* Build the AppendGraphExecute list once (binds this shape's fixed x_mem/y_mem, closed = re-runnable). */
    if (!s->exec_ready) {
        if (npu_list_create(&p->d, &s->exec_list) != 0 || npu_list_graph_exec(&s->g, s->exec_list) != 0) {
            p->failed = 1; *st = HPI_EDEVICE; return NULL;
        }
        s->exec_ready = 1;
        if (p->profile) p->profile->command_records++;
    } else {
        if (p->profile) p->profile->list_reuses++;
    }
    if (p->profile) {
        const double elapsed = npu_now_ms() - t0;
        p->profile->prepare_ms += elapsed;
        p->profile->record_ms += elapsed;
    }
    prepared->exec_list = s->exec_list;
    return s;
}

/* Read back one prepared op's output (device precision -> host f32) after its graph has executed. */
static hpi_status npu_finish(npu3720_priv *p, npu3720_prepared *prepared, const hpi_q8_0_gemm *op) {
    npu3720_shape *s = prepared->shape;
    const npu_graph_arg *ay = &s->g.arg[s->io.arg_y];
    const double t0 = p->profile ? npu_now_ms() : 0.0;
    const hpi_status st = stage_out_f32(op->y, ay->precision, prepared->y_mem,
                                       (size_t)op->M * (size_t)op->N, p->simd);
    if (p->profile) p->profile->output_ms += npu_now_ms() - t0;
    if (prepared->record) prepared->record->state = HPI3720_IDLE;
    else s->legacy_state = HPI3720_IDLE;
    memset(prepared, 0, sizeof *prepared);
    return st;
}

/* Uncached op -> the CPU reference (correct, not accelerated). Warns once. */
static hpi_status npu_cpu_fallback(hpi_device *dev, const hpi_q8_0_gemm *op) {
    static int said = 0;
    if (!said) { said = 1; fprintf(stderr, "hpi-3720: NOTE — some Q8_0 mul_mats have no DPU blob yet "
        "(uncached shape/tensor); those run on the CPU reference. Cached ops use the DPU.\n"); }
    if (getenv("GGML_NPU_VERBOSE")) fprintf(stderr, "hpi-3720: CPU-ref fallback op M=%lld N=%lld K=%lld\n",
        (long long) op->M, (long long) op->N, (long long) op->K);   /* diagnose which shapes lack a cached blob */
    const double t0 = dev->profile.enabled ? npu_now_ms() : 0.0;
    const hpi_status st = hpi_backend_cpu()->gemm(dev, op);
    if (dev->profile.enabled) {
        dev->profile.fallback_ms += npu_now_ms() - t0;
        dev->profile.fallback_ops++;
    }
    return st;
}

/* API wall times include driver overhead; sync_ms is not an isolated device compute timer. */
static hpi_status npu_submit_work(npu3720_priv *p, ze_command_list_handle_t *lists,
        uint32_t n, hpi_host_work work, void *user) {
    for (int i = 0; i < NPU3720_WEIGHT_MAX; ++i) if (p->cache[i].valid) {
        npu3720_shape *shape = &p->cache[i];
        for (uint32_t k = 0; k < n; ++k) {
            if (!shape->program && shape->exec_list == lists[k]) shape->legacy_state = HPI3720_SUBMITTED;
            for (int j = 0; shape->program && j < 2; ++j)
                if (shape->records[j].list == lists[k]) shape->records[j].state = HPI3720_SUBMITTED;
        }
    }
    double t0 = p->profile ? npu_now_ms() : 0.0;
    ze_result_t result = p->z.zeCommandQueueExecuteCommandLists(p->q, n, lists, NULL);
    if (p->profile) {
        p->profile->submit_ms += npu_now_ms() - t0;
        p->profile->submissions++;
        p->profile->graphs += n;
    }
    if (result != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "hpi-3720: queue submit failed: 0x%x\n", (unsigned)result);
        p->failed = 1; return HPI_EDEVICE;
    }
    if (work) {
        t0 = p->profile ? npu_now_ms() : 0.0;
        work(user);
        if (p->profile) {
            p->profile->host_work_ms += npu_now_ms() - t0;
            p->profile->host_work_calls++;
        }
    }
    t0 = p->profile ? npu_now_ms() : 0.0;
    result = p->z.zeCommandQueueSynchronize(p->q, UINT64_MAX);
    if (p->profile) p->profile->sync_ms += npu_now_ms() - t0;
    if (result != ZE_RESULT_SUCCESS) {
        fprintf(stderr, "hpi-3720: queue sync failed: 0x%x\n", (unsigned)result);
        p->failed = 1; return HPI_EDEVICE;
    }
    return HPI_OK;
}

static hpi_status npu_submit(npu3720_priv *p, ze_command_list_handle_t *lists, uint32_t n) {
    return npu_submit_work(p, lists, n, NULL, NULL);
}

static hpi_status npu_gemm(hpi_device *dev, const hpi_q8_0_gemm *op) {
    if (!dev || !dev->priv || !op) return HPI_EINVAL;
    npu3720_priv *p = (npu3720_priv *)dev->priv;
    hpi_status st;
    npu3720_prepared prepared;
    npu3720_shape *s = npu_prepare(p, op, &st, &prepared);
    if (!s) return st == HPI_UNAVAILABLE ? npu_cpu_fallback(dev, op) :
                   (st == NPU3720_RETRY_PREPARE ? HPI_EDEVICE : st);
    st = npu_submit(p, &prepared.exec_list, 1);
    if (st != HPI_OK) return st;
    return npu_finish(p, &prepared, op);
}

/* Batch of N INDEPENDENT ops: prepare all (stage inputs, one-time init), submit every graph in ONE
 * zeCommandQueueExecuteCommandLists + ONE zeCommandQueueSynchronize (amortizes the per-op submit+sync
 * that dominates small ops), then read all outputs. Uncached ops fall to the CPU reference in place. */
#define NPU_GEMM_BATCH_MAX 32
static hpi_status npu_gemm_batch_impl(hpi_device *dev, const hpi_q8_0_gemm *ops, int n,
        hpi_status *results, hpi_host_work work, void *user) {
    if (!dev || !dev->priv || !ops || n <= 0) return HPI_EINVAL;
    npu3720_priv *p = (npu3720_priv *)dev->priv;
    if (n > NPU_GEMM_BATCH_MAX) {                                   /* split oversized batches */
        for (int off = 0; off < n; off += NPU_GEMM_BATCH_MAX) {
            int m = (n - off < NPU_GEMM_BATCH_MAX) ? (n - off) : NPU_GEMM_BATCH_MAX;
            hpi_status st = npu_gemm_batch_impl(dev, ops + off, m, results ? results + off : NULL,
                    off + m == n ? work : NULL, user);
            if (st != HPI_OK) return st;
        }
        return HPI_OK;
    }
    npu3720_shape *sh[NPU_GEMM_BATCH_MAX] = {0};
    npu3720_prepared prepared[NPU_GEMM_BATCH_MAX] = {0};
    ze_command_list_handle_t lists[NPU_GEMM_BATCH_MAX];
    int nlist = 0, begin = 0;
    for (int i = 0; i < n; ++i) {
        hpi_status st;
        sh[i] = npu_prepare(p, &ops[i], &st, &prepared[i]);
        if (!sh[i] && st == NPU3720_RETRY_PREPARE) {
            if (!nlist) return HPI_EDEVICE;
            st = npu_submit(p, lists, (uint32_t)nlist);
            if (st != HPI_OK) return st;
            for (int j = begin; j < i; ++j) if (sh[j]) {
                st = npu_finish(p, &prepared[j], &ops[j]);
                if (st != HPI_OK) return st;
                if (results) results[j] = HPI_OK;
                sh[j] = NULL;
            }
            nlist = 0; begin = i;
            sh[i] = npu_prepare(p, &ops[i], &st, &prepared[i]);
        }
        if (!sh[i]) {
            if (st != HPI_UNAVAILABLE) return st == NPU3720_RETRY_PREPARE ? HPI_EDEVICE : st;
            if (results) { results[i] = HPI_UNAVAILABLE; continue; }
            st = npu_cpu_fallback(dev, &ops[i]);
            if (st != HPI_OK) return st;
            continue;
        }
        lists[nlist++] = prepared[i].exec_list;
    }
    if (nlist) {
        const hpi_status st = npu_submit_work(p, lists, (uint32_t)nlist, work, user);
        if (st != HPI_OK) return st;
    }
    if (!nlist && work) work(user);
    for (int i = begin; i < n; ++i) if (sh[i]) {
        const hpi_status st = npu_finish(p, &prepared[i], &ops[i]);
        if (st != HPI_OK) return st;
        if (results) results[i] = HPI_OK;
    }
    return HPI_OK;
}

static hpi_status npu_gemm_batch(hpi_device *dev, const hpi_q8_0_gemm *ops, int n) {
    return npu_gemm_batch_impl(dev, ops, n, NULL, NULL, NULL);
}

static hpi_status npu_gemm_batch_try(hpi_device *dev, const hpi_q8_0_gemm *ops, int n, hpi_status *results) {
    return npu_gemm_batch_impl(dev, ops, n, results, NULL, NULL);
}

static hpi_status npu_gemm_batch_overlap(hpi_device *dev, const hpi_q8_0_gemm *ops,
        int n, hpi_status *results, hpi_host_work work, void *user) {
    return npu_gemm_batch_impl(dev, ops, n, results, work, user);
}

static const hpi_backend_ops g_npu_ops = {
    .kind = HPI_BACKEND_NPU_3720,
    .name = "npu-3720",
    .is_hw = 1,
    .available = npu_available,
    .open = npu_open,
    .gemm = npu_gemm,
    .gemm_batch = npu_gemm_batch,
    .close = npu_close,
    .gemm_batch_try = npu_gemm_batch_try,
    .gemm_batch_overlap = npu_gemm_batch_overlap,
};

const hpi_backend_ops *hpi_backend_npu_3720(void) { return &g_npu_ops; }

#else  /* !HPI_HAVE_NPU_3720 — non-hardware build: no NPU backend at all */

const hpi_backend_ops *hpi_backend_npu_3720(void) { return (const hpi_backend_ops *)0; }

#endif
