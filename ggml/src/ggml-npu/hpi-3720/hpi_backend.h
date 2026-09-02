/*
 * hpi_backend.h — internal seam between the HPI dispatcher (hpi.c) and a concrete backend
 * (hpi_cpu.c, hpi_npu_3720.c). Not a public header. Each backend is a small explicit vtable;
 * no globals, the device carries its own ops and private state.
 */
#ifndef HPI_BACKEND_H
#define HPI_BACKEND_H

#include "hpi.h"

typedef struct hpi_backend_ops hpi_backend_ops;

struct hpi_device {
    const hpi_backend_ops *ops;   /* dispatch table for this device */
    void                  *priv;  /* backend-private state (owned by the backend) */
    hpi_device_info        info;
};

struct hpi_backend_ops {
    hpi_backend_kind kind;
    const char      *name;
    int              is_hw;
    /* 1 if this backend can be opened on this build/platform right now. */
    int        (*available)(void);
    /* Allocate priv + fill it; return HPI_OK or an error. */
    hpi_status (*open)(hpi_device *dev);
    /* Run one validated Q8_0 GEMM (shapes already checked by the dispatcher). */
    hpi_status (*gemm)(hpi_device *dev, const hpi_q8_0_gemm *op);
    /* Free priv. */
    void       (*close)(hpi_device *dev);
};

/* Backend factories (defined in their .c files). Return NULL if compiled out. */
const hpi_backend_ops *hpi_backend_cpu(void);
const hpi_backend_ops *hpi_backend_npu_3720(void);

#endif /* HPI_BACKEND_H */
