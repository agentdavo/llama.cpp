/*
 * hpi_npu3720_blob.h — the compute-seam contract shared by hpi_npu_3720.c (the backend) and
 * hpi_npu3720_blob.c (the blob provider compiled on the box). Keeps the io struct in one place so
 * both TUs agree on its layout.
 */
#ifndef HPI_NPU3720_BLOB_H
#define HPI_NPU3720_BLOB_H

#include "hpi.h"   /* hpi_status, hpi_q8_0_gemm */

/* How build_blob's blob binds its runtime graph arguments (see hpi_npu_3720.c shape_build). */
typedef struct {
    int arg_x;   /* graph argument index that receives activations X (>= 0)      */
    int arg_y;   /* graph argument index that produces output Y (>= 0)           */
    int arg_w;   /* weights input index, or < 0 if weights are baked into the blob */
    uint8_t *weight_image;       /* optional malloc-owned immutable image; caller frees */
    size_t weight_image_bytes;
} hpi_npu3720_blob_io;

/* Produce (or fetch from a cache) a native VPU-3720 blob computing Y[MxN] = X[MxK] . dequant(W)^T for
 * this exact shape+weights. On success: *blob = malloc'd blob (caller frees), *blob_len its size, *io
 * filled. Returns HPI_UNAVAILABLE when no blob is available for this op (caller falls back to CPU).
 * Declared only when a provider TU (hpi_npu3720_blob.c) is compiled in; otherwise hpi_npu_3720.c
 * supplies a local static stub. */
#if defined(HPI_NPU3720_BLOB_READY)
hpi_status hpi_npu3720_build_blob(const hpi_q8_0_gemm *op,
                                  uint8_t **blob, size_t *blob_len,
                                  hpi_npu3720_blob_io *io);
#endif

#endif /* HPI_NPU3720_BLOB_H */
