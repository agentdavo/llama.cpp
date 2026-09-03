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
| `hpi-3720/` GGUF reader + offload harness | **done** — reads Q8_0 tensors from a `.gguf`, runs them through the offload |
| `ggml-npu.cpp` host backend glue | **done** — registers via `ggml_add_backend(NPU)`; llama enumerates it as an `ACCEL` device |
| `hpi-3720/` NPU-3720 hardware path | **stub** — waits on the NCE descriptor / Q8_0 schedule (see outer CLAUDE.md "long game") |

First target op is **Q8_0 offload** (a mul_mat whose weight operand is Q8_0-quantized), because Q8_0
is a clean, well-understood block format and a natural fit for the NPU's INT8 MAC array.

## Build & verify

```sh
cmake -S . -B build -DGGML_NPU=ON        # from the llama.cpp root
cmake --build build --target test-backend-ops -j
./build/bin/test-backend-ops test -b NPU -o MUL_MAT
```

`test-backend-ops` enumerates every registered backend (so it confirms the NPU device is detected)
and runs the op suite against it. The NPU backend declines every op except Q8_0 `mul_mat`, and passes
all Q8_0 `mul_mat` cases — batched and GQA-broadcast included — against ggml's own reference. Since
the compute runs on the hpi CPU reference, results are correct today; a `-DGGML_NPU_HW=ON` build will
route them to silicon with no change to this backend.

The device shows up as an `ACCEL` named `hpi-3720` (reg family `NPU`, mirroring Hexagon's
`HTP`/`HTP0` split), so it is selectable and an offload target:

```sh
llama-cli --list-devices                 # -> hpi-3720: Intel NPU 2.7 / VPU 3720 (...)
llama-cli --device hpi-3720 -ngl 999 -m model-q8_0.gguf -p "..."
```

`--device` matches the name case-insensitively. Q8_0 weight matmuls the device accepts get
offloaded; everything else falls back to CPU. (`-ncmoe`/`-cmoe` pin MoE experts to CPU, so keep
them low/unset if you want expert matmuls to reach `hpi-3720`.) Note this is still the CPU
reference — selecting it is correctness/plumbing, not speed, until the silicon path lands.

### Prove it computes (no download)

Synthesize a tiny valid Q8_0 llama model, then run it and watch the backend log its work:

```sh
python3 ggml/src/ggml-npu/hpi-3720/make_tiny_q8_0_gguf.py /tmp/tiny-q8_0.gguf   # needs numpy
GGML_NPU_VERBOSE=1 ./build/bin/llama-bench -m /tmp/tiny-q8_0.gguf -ngl 999 --device hpi-3720 -p 16 -n 8
```

`llama-bench` drives the model with random tokens (no tokenizer needed) and prints, e.g.:

```
hpi-3720: computing Q8_0 mul_mat on the NPU backend (first op 'Qcur-0': M=16 N=64 K=64)
| llama ?B Q8_0 | ... | NPU | 999 | hpi-3720 | pp16 |  12535 ± 954 |
hpi-3720: backend ran 90 Q8_0 mul_mat op(s), 0.011 GFLOP total
```

`Qcur-0` is the layer-0 query projection — a real transformer weight matmul routed to and computed
by this backend. Set `GGML_NPU_VERBOSE=1` to get the per-run counts (the ggml info log itself is
suppressed by `llama-bench`); without it the backend is silent. The model's weights are random, so
the *output* is meaningless — this proves the offload path executes, not model quality.

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
