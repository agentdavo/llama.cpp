/*
 * hpi_npu_3720.c — the real Intel NPU 2.7 / VPU 3720 backend.
 *
 * STATUS: STUB. The interface is wired; the hardware path is not. When HPI_HAVE_NPU_3720 is defined
 * (a Windows x64 build with the NPU present and the src/ direct-access toolkit on the include path),
 * this compiles the real path; otherwise it compiles to an "unavailable" backend so callers fall
 * back to the CPU reference and the module still builds cross-platform.
 *
 * The real path, when implemented, will follow the verified src/ ladder (src/ze_npu*.c, src/npu.h):
 *   1. npu_ze_load + npu_dev_open                 (Level Zero, select the VPU)
 *   2. build/load a Q8_0-GEMM schedule blob       (re/emit_*.py analog for the NCE/DPU MAC array)
 *   3. npu_mem_alloc NPU-visible buffers          (stage W, X in; Y out)
 *   4. npu_graph_create -> init -> exec           (VPU_CMD_INFERENCE_EXECUTE)
 *   5. read back Y and return
 * Until steps 2-4 exist (the NCE descriptor layout is not yet mapped — see the outer CLAUDE.md
 * "long game"), this backend must NOT claim to run: verify-twice discipline, never present a guess
 * as a result.
 */
#include "hpi_backend.h"

#if defined(HPI_HAVE_NPU_3720)

/*
 * Hardware build. This TU expects the src/ toolkit (npu.h) reachable and NPU_NO_GDN defined by the
 * build. Left deliberately unimplemented rather than faked: opening succeeds only once a real Q8_0
 * schedule exists, so for now the backend still advertises unavailable even on hardware.
 */
static int  npu_available(void) { return 0; }   /* flip to 1 when steps 2-4 above are real */
static hpi_status npu_open(hpi_device *dev) { (void)dev; return HPI_UNAVAILABLE; }
static hpi_status npu_gemm(hpi_device *dev, const hpi_q8_0_gemm *op) { (void)dev; (void)op; return HPI_UNAVAILABLE; }
static void npu_close(hpi_device *dev) { (void)dev; }

static const hpi_backend_ops g_npu_ops = {
    .kind = HPI_BACKEND_NPU_3720,
    .name = "npu-3720",
    .is_hw = 1,
    .available = npu_available,
    .open = npu_open,
    .gemm = npu_gemm,
    .close = npu_close,
};

const hpi_backend_ops *hpi_backend_npu_3720(void) { return &g_npu_ops; }

#else  /* !HPI_HAVE_NPU_3720 — non-hardware build: no NPU backend at all */

const hpi_backend_ops *hpi_backend_npu_3720(void) { return (const hpi_backend_ops *)0; }

#endif
