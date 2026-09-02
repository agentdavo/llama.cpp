/*
 * hpi_gguf.c — minimal read-only GGUF parser (see hpi_gguf.h). Cursor-based with bounds checks on
 * every read; any overrun or malformed field fails the open rather than reading out of bounds.
 */
#include "hpi_gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGUF_MAGIC 0x46554747u  /* "GGUF" little-endian */

/* GGUF metadata value types */
enum { GT_U8=0, GT_I8, GT_U16, GT_I16, GT_U32, GT_I32, GT_F32, GT_BOOL, GT_STR, GT_ARR, GT_U64, GT_I64, GT_F64 };

typedef struct { const uint8_t *p, *end; int err; } cur;

static uint32_t rd_u32(cur *c) {
    if (c->err || c->p + 4 > c->end) { c->err = 1; return 0; }
    uint32_t v; memcpy(&v, c->p, 4); c->p += 4; return v;
}
static uint64_t rd_u64(cur *c) {
    if (c->err || c->p + 8 > c->end) { c->err = 1; return 0; }
    uint64_t v; memcpy(&v, c->p, 8); c->p += 8; return v;
}
/* string = u64 len + bytes; returns pointer to bytes (not null-terminated) and its length */
static const uint8_t *rd_str(cur *c, uint64_t *len) {
    uint64_t n = rd_u64(c);
    if (c->err || c->p + n > c->end || n > (uint64_t)(c->end - c->p)) { c->err = 1; *len = 0; return NULL; }
    const uint8_t *s = c->p; c->p += n; *len = n; return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
        case GT_U8: case GT_I8: case GT_BOOL:            return 1;
        case GT_U16: case GT_I16:                        return 2;
        case GT_U32: case GT_I32: case GT_F32:           return 4;
        case GT_U64: case GT_I64: case GT_F64:           return 8;
        default:                                         return 0; /* STR/ARR handled separately */
    }
}

/* advance the cursor past one value of the given type (used to skip metadata we don't interpret) */
static void skip_value(cur *c, uint32_t type) {
    if (c->err) return;
    if (type == GT_STR) { uint64_t l; rd_str(c, &l); return; }
    if (type == GT_ARR) {
        uint32_t et = rd_u32(c);
        uint64_t n  = rd_u64(c);
        if (c->err) return;
        if (et == GT_STR) { for (uint64_t i = 0; i < n && !c->err; i++) { uint64_t l; rd_str(c, &l); } return; }
        size_t es = scalar_size(et);
        if (es == 0) { c->err = 1; return; }         /* array of array/unknown: unsupported */
        if (c->p + n * es > c->end) { c->err = 1; return; }
        c->p += n * es;
        return;
    }
    size_t s = scalar_size(type);
    if (s == 0 || c->p + s > c->end) { c->err = 1; return; }
    c->p += s;
}

/* bytes of tensor data for a ggml type + dims; 0 on unsupported type */
static uint64_t tensor_nbytes(int32_t type, const uint64_t *ne, uint32_t nd) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < nd; i++) n *= ne[i];
    switch (type) {
        case HPI_GGML_TYPE_F32:  return n * 4;
        case HPI_GGML_TYPE_F16:  return n * 2;
        case HPI_GGML_TYPE_Q8_0: return (ne[0] % 32 == 0) ? (n / 32) * 34 : 0;
        default:                 return 0;  /* other quant types: size unknown to this minimal reader */
    }
}

int hpi_gguf_open(const char *path, hpi_gguf *g) {
    memset(g, 0, sizeof *g);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "hpi_gguf: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fprintf(stderr, "hpi_gguf: empty/bad file\n"); fclose(f); return -1; }
    g->file = (uint8_t *)malloc((size_t)sz);
    if (!g->file) { fclose(f); return -1; }
    if (fread(g->file, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "hpi_gguf: short read\n"); fclose(f); free(g->file); return -1; }
    fclose(f);
    g->file_size = (size_t)sz;

    cur c = { g->file, g->file + g->file_size, 0 };
    if (rd_u32(&c) != GGUF_MAGIC) { fprintf(stderr, "hpi_gguf: not a GGUF file\n"); goto fail; }
    uint32_t version = rd_u32(&c);
    if (version < 2 || version > 3) { fprintf(stderr, "hpi_gguf: unsupported version %u\n", version); goto fail; }
    uint64_t n_tensors = rd_u64(&c);
    uint64_t n_kv      = rd_u64(&c);
    if (c.err || n_tensors > (1u << 24)) { fprintf(stderr, "hpi_gguf: bad header\n"); goto fail; }

    g->alignment = 32;  /* GGUF default */
    for (uint64_t i = 0; i < n_kv && !c.err; i++) {
        uint64_t klen; const uint8_t *key = rd_str(&c, &klen);
        uint32_t vtype = rd_u32(&c);
        if (c.err) break;
        if (klen == 17 && key && memcmp(key, "general.alignment", 17) == 0 && vtype == GT_U32) {
            uint32_t a = rd_u32(&c);
            if (a && (a & (a - 1)) == 0) g->alignment = a;  /* power of two */
        } else {
            skip_value(&c, vtype);
        }
    }
    if (c.err) { fprintf(stderr, "hpi_gguf: metadata parse error\n"); goto fail; }

    g->tensors = (hpi_gguf_tensor *)calloc(n_tensors ? n_tensors : 1, sizeof *g->tensors);
    if (!g->tensors) goto fail;

    for (uint64_t i = 0; i < n_tensors && !c.err; i++) {
        hpi_gguf_tensor *t = &g->tensors[i];
        uint64_t nlen; const uint8_t *nm = rd_str(&c, &nlen);
        if (c.err || nlen >= sizeof t->name) { c.err = 1; break; }
        memcpy(t->name, nm, nlen); t->name[nlen] = 0;
        t->n_dims = rd_u32(&c);
        if (t->n_dims > 4) { c.err = 1; break; }
        t->ne[0] = t->ne[1] = t->ne[2] = t->ne[3] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(&c);
        t->type   = (int32_t)rd_u32(&c);
        t->nbytes = 0;                 /* filled after we know the data base */
        uint64_t off = rd_u64(&c);
        t->data = (const uint8_t *)(uintptr_t)off;  /* stash offset; rebased below */
    }
    if (c.err) { fprintf(stderr, "hpi_gguf: tensor-info parse error\n"); goto fail; }

    /* data section begins at the next alignment boundary after the tensor infos */
    {
        size_t pos = (size_t)(c.p - g->file);
        size_t data_base = (pos + g->alignment - 1) & ~((size_t)g->alignment - 1);
        for (uint64_t i = 0; i < n_tensors; i++) {
            hpi_gguf_tensor *t = &g->tensors[i];
            uint64_t off = (uint64_t)(uintptr_t)t->data;
            t->nbytes = tensor_nbytes(t->type, t->ne, t->n_dims);
            if (data_base + off + t->nbytes > g->file_size) {
                fprintf(stderr, "hpi_gguf: tensor '%s' data out of range\n", t->name);
                goto fail;
            }
            t->data = g->file + data_base + off;
        }
    }

    g->n_tensors = (uint32_t)n_tensors;
    return 0;

fail:
    hpi_gguf_close(g);
    return -1;
}

void hpi_gguf_close(hpi_gguf *g) {
    if (!g) return;
    free(g->tensors);
    free(g->file);
    memset(g, 0, sizeof *g);
}

const hpi_gguf_tensor *hpi_gguf_find(const hpi_gguf *g, const char *name) {
    for (uint32_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    return NULL;
}
