/*
 * hpi.c — the HPI dispatcher. Shape validation + backend selection; the actual math lives in the
 * backends. No hidden state: every function takes the device (or a kind) explicitly.
 */
#include <stdlib.h>
#include "hpi_backend.h"

/* Portable CPU reference: keep its accumulation order as the numerical oracle. */
static int cpu_available(void) { return 1; }   /* always */

static hpi_status cpu_open(hpi_device *dev) { (void)dev; return HPI_OK; }
static void       cpu_close(hpi_device *dev) { (void)dev; }

static hpi_status cpu_gemm(hpi_device *dev, const hpi_q8_0_gemm *op) {
    (void)dev;
    const int64_t M = op->M, N = op->N, K = op->K;
    const int64_t kb = K / HPI_QK8_0;             /* blocks per row */

    for (int64_t m = 0; m < M; m++) {
        const float *xr = op->x + m * K;
        float       *yr = op->y + m * N;
        for (int64_t n = 0; n < N; n++) {
            const hpi_block_q8_0 *wr = op->w + n * kb;
            float acc = 0.0f;
            for (int64_t b = 0; b < kb; b++) {
                const float d = hpi_f16_to_f32(wr[b].d);
                const float *xk = xr + b * HPI_QK8_0;
                float bacc = 0.0f;
                for (int i = 0; i < HPI_QK8_0; i++) {
                    bacc += (float)wr[b].qs[i] * xk[i];
                }
                acc += bacc * d;                  /* scale per block, like ggml vec_dot_q8_0 */
            }
            yr[n] = acc;
        }
    }
    return HPI_OK;
}

static const hpi_backend_ops g_cpu_ops = {
    .kind = HPI_BACKEND_CPU,
    .name = "cpu-reference",
    .is_hw = 0,
    .available = cpu_available,
    .open = cpu_open,
    .gemm = cpu_gemm,
    .close = cpu_close,
};

const hpi_backend_ops *hpi_backend_cpu(void) { return &g_cpu_ops; }


/* Ordered by preference for HPI_BACKEND_AUTO: hardware first, CPU reference last. */
static const hpi_backend_ops *resolve(hpi_backend_kind kind) {
    const hpi_backend_ops *npu = hpi_backend_npu_3720();
    const hpi_backend_ops *cpu = hpi_backend_cpu();
    switch (kind) {
        case HPI_BACKEND_NPU_3720: return npu;
        case HPI_BACKEND_CPU:      return cpu;
        case HPI_BACKEND_AUTO:
        default:
            if (npu && npu->available && npu->available()) return npu;
            return cpu;
    }
}

int hpi_backend_available(hpi_backend_kind kind) {
    const hpi_backend_ops *ops = resolve(kind);
    return ops && ops->available && ops->available();
}

hpi_device *hpi_open(hpi_backend_kind kind, hpi_status *status) {
    hpi_status st = HPI_OK;
    const hpi_backend_ops *ops = resolve(kind);
    if (!ops || !ops->available || !ops->available()) {
        st = HPI_UNAVAILABLE;
        goto out;
    }
    hpi_device *dev = (hpi_device *)calloc(1, sizeof *dev);
    if (!dev) { st = HPI_ENOMEM; goto out; }
    dev->ops       = ops;
    dev->priv      = NULL;
    dev->info.kind = ops->kind;
    dev->info.name = ops->name;
    dev->info.is_hw = ops->is_hw;
    const char *profile = getenv("GGML_NPU_PROFILE");
    dev->profile.enabled = profile && profile[0] && profile[0] != '0';
    st = ops->open ? ops->open(dev) : HPI_OK;
    if (st != HPI_OK) { free(dev); dev = NULL; }
    if (status) *status = st;
    return dev;
out:
    if (status) *status = st;
    return NULL;
}

hpi_status hpi_get_info(const hpi_device *dev, hpi_device_info *out) {
    if (!dev || !out) return HPI_EINVAL;
    *out = dev->info;
    return HPI_OK;
}

hpi_status hpi_get_profile(const hpi_device *dev, hpi_profile *out) {
    if (!dev || !out) return HPI_EINVAL;
    *out = dev->profile;
    return HPI_OK;
}

hpi_status hpi_q8_0_gemm_run(hpi_device *dev, const hpi_q8_0_gemm *op) {
    if (!dev || !dev->ops || !dev->ops->gemm || !op) return HPI_EINVAL;
    if (!op->w || !op->x || !op->y)                  return HPI_EINVAL;
    if (op->M <= 0 || op->N <= 0 || op->K <= 0)      return HPI_EINVAL;
    if (op->K % HPI_QK8_0 != 0)                      return HPI_EINVAL;
    return dev->ops->gemm(dev, op);
}

hpi_status hpi_q8_0_gemm_batch(hpi_device *dev, const hpi_q8_0_gemm *ops, int n) {
    if (!dev || !dev->ops || !ops || n <= 0) return HPI_EINVAL;
    for (int i = 0; i < n; i++) {
        const hpi_q8_0_gemm *op = &ops[i];
        if (!op->w || !op->x || !op->y)             return HPI_EINVAL;
        if (op->M <= 0 || op->N <= 0 || op->K <= 0) return HPI_EINVAL;
        if (op->K % HPI_QK8_0 != 0)                 return HPI_EINVAL;
    }
    if (dev->ops->gemm_batch) return dev->ops->gemm_batch(dev, ops, n);
    if (!dev->ops->gemm)      return HPI_EINVAL;
    for (int i = 0; i < n; i++) {              /* no batch path -> run one at a time */
        hpi_status st = dev->ops->gemm(dev, &ops[i]);
        if (st != HPI_OK) return st;
    }
    return HPI_OK;
}

void hpi_close(hpi_device *dev) {
    if (!dev) return;
    if (dev->ops && dev->ops->close) dev->ops->close(dev);
    free(dev);
}

hpi_status hpi_q8_0_gemm_batch_try(hpi_device *dev, const hpi_q8_0_gemm *ops, int n, hpi_status *results) {
    if (!dev || !dev->ops || !ops || !results || n <= 0) return HPI_EINVAL;
    for (int i = 0; i < n; i++) {
        const hpi_q8_0_gemm *op = &ops[i];
        if (!op->w || !op->x || !op->y || op->M <= 0 || op->N <= 0 || op->K <= 0 || op->K % HPI_QK8_0) return HPI_EINVAL;
        results[i] = HPI_UNAVAILABLE;
    }
    if (!dev->ops->gemm_batch_try) return HPI_OK;
    return dev->ops->gemm_batch_try(dev, ops, n, results);
}

hpi_status hpi_q8_0_gemm_batch_overlap(hpi_device *dev, const hpi_q8_0_gemm *ops,
        int n, hpi_status *results, hpi_host_work work, void *user) {
    if (!dev || !dev->ops || !ops || !results || n <= 0) return HPI_EINVAL;
    for (int i = 0; i < n; ++i) {
        const hpi_q8_0_gemm *op = &ops[i];
        if (!op->w || !op->x || !op->y || op->M <= 0 || op->N <= 0 || op->K <= 0 || op->K % HPI_QK8_0) return HPI_EINVAL;
        results[i] = HPI_UNAVAILABLE;
    }
    if (dev->ops->gemm_batch_overlap)
        return dev->ops->gemm_batch_overlap(dev, ops, n, results, work, user);
    const hpi_status st = hpi_q8_0_gemm_batch_try(dev, ops, n, results);
    if (st == HPI_OK && work) work(user);
    return st;
}

const char *hpi_status_str(hpi_status s) {
    switch (s) {
        case HPI_OK:          return "ok";
        case HPI_UNAVAILABLE: return "backend unavailable on this build/platform";
        case HPI_EINVAL:      return "invalid argument";
        case HPI_ENOMEM:      return "out of memory";
        case HPI_EDEVICE:     return "device/driver rejected the job";
        default:              return "unknown status";
    }
}
