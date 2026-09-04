#ifndef GGML_NPU_XE_LPG_H
#define GGML_NPU_XE_LPG_H

struct ggml_tensor;
struct ggml_backend_npu_xe_lpg;

ggml_backend_npu_xe_lpg *ggml_backend_npu_xe_lpg_create(const void *model_identity);
void ggml_backend_npu_xe_lpg_destroy(ggml_backend_npu_xe_lpg *bridge);
int ggml_backend_npu_xe_lpg_role(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node);
void ggml_backend_npu_xe_lpg_capture(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node, int role);
void ggml_backend_npu_xe_lpg_complete(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down);

#endif
