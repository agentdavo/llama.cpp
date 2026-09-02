#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// Intel NPU 2.7 / VPU 3720 backend ("AI Boost", PCI 8086:AD1D).
//
// Currently a "fake" NPU: it registers as an accelerator device so llama enumerates and can offload
// to it, but the compute runs through the cross-platform hpi-3720 CPU reference (correct, not fast).
// The real VPU-3720 path (src/ direct-access toolkit) is stubbed behind HPI_HAVE_NPU_3720. This lets
// the backend, device enumeration, and offload wiring be developed and tested with no NPU present.
// First offloaded op is Q8_0 mul_mat.

GGML_BACKEND_API ggml_backend_t     ggml_backend_npu_init(void);
GGML_BACKEND_API bool               ggml_backend_is_npu(ggml_backend_t backend);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_npu_reg(void);

#ifdef  __cplusplus
}
#endif
