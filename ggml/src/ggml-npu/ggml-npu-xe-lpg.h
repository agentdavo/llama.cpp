#ifndef GGML_NPU_XE_LPG_H
#define GGML_NPU_XE_LPG_H

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend_npu_xe_lpg;

ggml_backend_npu_xe_lpg *ggml_backend_npu_xe_lpg_create(const void *model_identity);
void ggml_backend_npu_xe_lpg_destroy(ggml_backend_npu_xe_lpg *bridge);
int ggml_backend_npu_xe_lpg_role(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node);
void ggml_backend_npu_xe_lpg_capture(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *node, int role);
int ggml_backend_npu_xe_lpg_begin_replace(
        ggml_backend_npu_xe_lpg *bridge, ggml_cgraph *graph, int gate_index);
int ggml_backend_npu_xe_lpg_replace(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down);
void ggml_backend_npu_xe_lpg_abort(ggml_backend_npu_xe_lpg *bridge);
void ggml_backend_npu_xe_lpg_complete(ggml_backend_npu_xe_lpg *bridge, const ggml_tensor *down);
void ggml_backend_npu_xe_lpg_report_graph(const ggml_backend_npu_xe_lpg *bridge);

#endif
