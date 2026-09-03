/*
 * hpi.h — Host-Platform Interface to the Intel NPU 2.7 / VPU 3720 ("AI Boost", PCI 8086:AD1D).
 *
 * This is the thin C ABI the ggml-npu host backend calls to offload a Q8_0 matmul. It abstracts
 * over WHO computes:
 *
 *   HPI_BACKEND_CPU       portable C reference — always available, the "cross platform for now" path.
 *   HPI_BACKEND_NPU_3720  the real NPU, driven through the src/ direct-access toolkit (src/npu.h:
 *                         Level Zero -> VPU_CMD_INFERENCE_EXECUTE -> NCE/DPU MAC array). Only built
 *                         with real behavior on Windows x64 with the NPU present (HPI_HAVE_NPU_3720);
 *                         elsewhere it reports HPI_UNAVAILABLE so callers fall back to CPU.
 *
 * Design follows the src/ Carmack discipline: state is explicit (the caller owns an hpi_device and
 * passes it in — no globals), every entry point returns a checked status, and the pure math (the
 * GEMM, the Q8_0 (de)quant) is separated from the effectful device path.
 *
 * Status: SCAFFOLD. The op contract and the CPU reference are real and tested; the NPU_3720 backend
 * is a stub until the JSM/DMA path (src/ze_npu*) is wired to a Q8_0 GEMM schedule. See README.md.
 */
#ifndef HPI_H
#define HPI_H

#include <stdint.h>
#include <stddef.h>
#include "hpi_q8_0.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    HPI_OK               =  0,
    HPI_UNAVAILABLE      = -1,  /* backend not present on this platform/build */
    HPI_EINVAL           = -2,  /* bad argument (null, shape, alignment) */
    HPI_ENOMEM           = -3,  /* allocation failed */
    HPI_EDEVICE          = -4   /* the accelerator or its driver rejected the job */
} hpi_status;

typedef enum {
    HPI_BACKEND_AUTO     = 0,   /* pick NPU_3720 if usable, else CPU */
    HPI_BACKEND_CPU      = 1,
    HPI_BACKEND_NPU_3720 = 2
} hpi_backend_kind;

/* Opaque device/context handle. One owner; created and destroyed explicitly. */
typedef struct hpi_device hpi_device;

typedef struct {
    hpi_backend_kind kind;      /* which backend actually backs this device */
    const char      *name;      /* human-readable, e.g. "cpu-reference" / "npu-3720" */
    int              is_hw;     /* 1 if a real accelerator, 0 if a software reference */
} hpi_device_info;

/*
 * Q8_0 GEMM offload — ggml mul_mat semantics.
 *
 *   weights W : N rows x K cols, Q8_0-quantized. Row-major blocks: W[n] is (K/HPI_QK8_0) contiguous
 *               hpi_block_q8_0, so `w` points at N*(K/HPI_QK8_0) blocks.
 *   input   X : M rows x K cols, float32, row-major (`x` = M*K floats).
 *   output  Y : M rows x N cols, float32, row-major (`y` = M*N floats), Y[m,n] = sum_k deq(W[n,k])*X[m,k].
 *
 * K must be a multiple of HPI_QK8_0. All buffers are caller-owned host memory; the backend copies
 * in/out as needed (the NPU path will stage them into NPU-visible memory).
 */
typedef struct {
    int64_t                 M, N, K;
    const hpi_block_q8_0   *w;  /* N * (K/HPI_QK8_0) blocks */
    const float            *x;  /* M * K floats */
    float                  *y;  /* M * N floats (output) */
} hpi_q8_0_gemm;

/* Is a given backend usable on this build/platform right now? (Cheap; no device open.) */
int         hpi_backend_available(hpi_backend_kind kind);

/* Open a device for the given backend (AUTO selects). Returns NULL and sets *status on failure. */
hpi_device *hpi_open(hpi_backend_kind kind, hpi_status *status);

/* Query which backend a device ended up using. */
hpi_status  hpi_get_info(const hpi_device *dev, hpi_device_info *out);

/* Run one Q8_0 GEMM. Validates shapes, then dispatches to the device's backend. */
hpi_status  hpi_q8_0_gemm_run(hpi_device *dev, const hpi_q8_0_gemm *op);

/* Run N INDEPENDENT Q8_0 GEMMs as a batch. On the NPU this stages all inputs, submits all N graphs in
 * ONE queue execute + ONE synchronize, then reads all outputs — amortizing the per-op submit+sync that
 * dominates small ops. The ops MUST be independent (none reads another's output). Validates each op;
 * falls back to running them one at a time if the backend has no batch path. */
hpi_status  hpi_q8_0_gemm_batch(hpi_device *dev, const hpi_q8_0_gemm *ops, int n);

/* Release a device. Safe on NULL. */
void        hpi_close(hpi_device *dev);

/* Human-readable status string (never NULL). */
const char *hpi_status_str(hpi_status s);

#ifdef __cplusplus
}
#endif

#endif /* HPI_H */
