/*
 * hpi_gguf.h — a tiny, read-only GGUF reader, just enough to enumerate tensors and reach their raw
 * bytes so the Q8_0 offload can be pointed at a real model file. Self-contained (no ggml), so the
 * hpi-3720 module stays dependency-free and testable anywhere. Little-endian hosts only (x86/ARM).
 *
 * It parses the header, skips all metadata KV pairs (every GGUF value type), reads the tensor infos,
 * and computes the aligned data section — it does not interpret metadata semantics beyond
 * general.alignment.
 */
#ifndef HPI_GGUF_H
#define HPI_GGUF_H

#include <stdint.h>
#include <stddef.h>

/* ggml_type values we care about (must match ggml-common.h / ggml.h enum) */
#define HPI_GGML_TYPE_F32  0
#define HPI_GGML_TYPE_F16  1
#define HPI_GGML_TYPE_Q8_0 8

typedef struct {
    char           name[256];
    int32_t        type;        /* ggml_type */
    uint32_t       n_dims;
    uint64_t       ne[4];       /* dims, ne[0] fastest */
    const uint8_t *data;        /* into the mapped/loaded file */
    uint64_t       nbytes;      /* size of this tensor's data */
} hpi_gguf_tensor;

typedef struct {
    uint8_t         *file;      /* whole file, owned */
    size_t           file_size;
    hpi_gguf_tensor *tensors;   /* owned array */
    uint32_t         n_tensors;
    uint32_t         alignment; /* general.alignment, default 32 */
} hpi_gguf;

/* Returns 0 on success, -1 on any parse/IO error (message on stderr). Fills *g (caller frees via close). */
int  hpi_gguf_open(const char *path, hpi_gguf *g);
void hpi_gguf_close(hpi_gguf *g);

/* Find a tensor by exact name, or NULL. */
const hpi_gguf_tensor *hpi_gguf_find(const hpi_gguf *g, const char *name);

#endif /* HPI_GGUF_H */
