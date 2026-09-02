# ggml-npu — Intel NPU 2.7 / VPU 3720 backend (work in progress)

A ggml backend for the Intel NPU ("AI Boost", NPU 2.7 / VPU 37xx, PCI `8086:AD1D`) that talks to
the device **directly** — no OpenVINO runtime — reusing the direct-access toolkit developed in the
[`agentdavo/npu`](https://github.com/agentdavo/npu) project (`src/npu.h`: Level Zero → JSM/DMA →
the NCE/DPU MAC array).

Structurally this mirrors `ggml-hexagon`: a host-side backend (`ggml-npu.cpp`, *not yet added*) plus
a device/interface layer. Here that layer is [`hpi-3720/`](hpi-3720/) — the **Host-Platform Interface
for VPU 3720**, the seam through which the host offloads compute to the accelerator.

## Status

| Piece | State |
|-------|-------|
| `hpi-3720/` HPI + Q8_0 GEMM contract | **done** — API + portable CPU reference, self-tested |
| `hpi-3720/` NPU-3720 hardware path | **stub** — waits on the NCE descriptor / Q8_0 schedule (see outer CLAUDE.md "long game") |
| `ggml-npu.cpp` host backend glue | **not started** — will register with `ggml_add_backend(NPU)` |

First target op is **Q8_0 offload** (a mul_mat whose weight operand is Q8_0-quantized), because Q8_0
is a clean, well-understood block format and a natural fit for the NPU's INT8 MAC array.

## Cross-platform for now

`hpi-3720/` is deliberately **standalone and cross-platform**: it builds on any C11 compiler with a
CPU reference backend and no ggml or CPU-feature dependencies, so the interface and its tests can be
developed and verified anywhere (including CI without an NPU). The real VPU-3720 path is guarded
behind `HPI_HAVE_NPU_3720` and only does work on a Windows x64 build with the NPU present; on every
other build the NPU backend reports "unavailable" and callers fall back to the CPU reference.

It is intentionally **not yet wired into the ggml backend registry** (`ggml/src/CMakeLists.txt`
`ggml_add_backend`), so it cannot break the main llama.cpp build. It gets added there once the
`ggml-npu.cpp` host glue and a real NPU path exist.

See [`hpi-3720/README.md`](hpi-3720/README.md) for the interface, the build/test steps, and the
implementation plan.
