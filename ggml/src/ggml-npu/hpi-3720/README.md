# hpi-3720 — Host-Platform Interface for the Intel NPU 2.7 (VPU 3720)

The thin C ABI the `ggml-npu` host backend calls to offload a **Q8_0 GEMM** to the Intel NPU. It
abstracts over *who computes* so the same call site runs on a portable CPU reference today and on the
real NPU once the hardware path exists.

This is the NPU analog of `ggml-hexagon/htp/` (the Hexagon DSP device layer): the host backend stays
thin, and the accelerator-specific work — memory staging, the compute schedule, the submit/sync — lives
behind this interface.

## Files

| File | Role |
|------|------|
| `hpi.h` | Public API: `hpi_open` / `hpi_q8_0_gemm_run` / `hpi_close`, `hpi_q8_0_gemm` op, status codes |
| `hpi_q8_0.h` | `hpi_block_q8_0` (byte-exact mirror of ggml `block_q8_0`, 34 B) + portable row quantizer |
| `hpi_fp16.h` | Portable IEEE-754 half↔float (software fallback; F16C on a HW build) |
| `hpi_backend.h` | Internal seam: the backend vtable (`hpi_backend_ops`) and the device struct |
| `hpi.c` | Dispatcher: shape validation + backend selection (AUTO → NPU if usable, else CPU) |
| `hpi_cpu.c` | Portable CPU reference backend — the golden reference the NPU path must match |
| `hpi_npu_3720.c` | The real VPU-3720 backend — **stub**, guarded by `HPI_HAVE_NPU_3720` |
| `hpi_gguf.h` / `hpi_gguf.c` | Minimal read-only GGUF parser (enumerate tensors, reach Q8_0 bytes) |
| `gguf_q8_0_offload.c` | Tool: point the offload at a real `.gguf` (`--list` / `--tensor` / `--rows` / `--check`) |
| `test/test_q8_0_gemm.c` | Deterministic GEMM self-test vs. an independent double-precision reference |
| `test/test_gguf.c` | Writes a synthetic Q8_0 GGUF, reads it back, and offloads it end to end |

## The op

`hpi_q8_0_gemm` follows ggml `mul_mat` semantics:

```
weights W : N × K, Q8_0-quantized   (N rows, each K/32 contiguous hpi_block_q8_0)
input   X : M × K, float32          (row-major)
output  Y : M × N, float32          Y[m,n] = Σ_k dequant(W[n,k]) · X[m,k]
```

`K` must be a multiple of 32 (`HPI_QK8_0`). All buffers are caller-owned host memory; a backend
copies/stages in and out as needed (the NPU path will move them into NPU-visible memory).

## Build & test (standalone, cross-platform)

```sh
cmake -S . -B build -DHPI3720_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Verified with gcc and clang under the project's strict set
(`-std=c11 -Wall -Wextra -Wpedantic -Werror -Wconversion -Wshadow -Wcast-align -Wpointer-arith`).
The tests use fixed seeds and fixed shapes and compare against a reference accumulated in `double`
over the *same* quantized weights (so only float-vs-double accumulation differs) — max relative error
is ~1e-4.

### Offload a real model

```sh
# once you have a Q8_0 GGUF (e.g. Qwen3.5-0.8B-Q8_0.gguf) on a machine that can reach Hugging Face:
./build/gguf_q8_0_offload model.gguf --list                 # list its Q8_0 tensors
./build/gguf_q8_0_offload model.gguf --tensor <name> --rows 32 --check
```

`test_gguf` leaves a small `hpi_synth_q8_0.gguf` you can feed to the tool without any download.

## Design notes (from the `agentdavo/npu` Carmack discipline)

- **State is explicit.** The caller owns an `hpi_device` and passes it in; no globals. Each backend
  carries its own ops table and private state.
- **Pure vs. effectful split.** The GEMM math and the Q8_0 (de)quant are pure and live in headers /
  `hpi_cpu.c`; device open/submit/readback (the effects) live in the backend.
- **Every entry point returns a checked status**; the dispatcher validates shapes before dispatch.
- **ABI is machine-checked.** `static_assert(sizeof(hpi_block_q8_0) == 34)` keeps the block in lock-step
  with ggml.
- **No unverified hardware claims.** `hpi_npu_3720.c` reports `HPI_UNAVAILABLE` until a real Q8_0
  schedule runs on silicon — it never fakes a result.

## Implementing the NPU-3720 path (next)

Build with `-DHPI3720_HAVE_NPU=ON` on Windows x64 with the NPU present, wire the `src/` toolkit
includes in `CMakeLists.txt`, then fill in `hpi_npu_3720.c` following the verified `src/` ladder:

1. `npu_ze_load` + `npu_dev_open` — select the VPU via Level Zero (`src/ze_npu*.c`).
2. Build/load a **Q8_0 GEMM schedule blob** for the NCE/DPU MAC array (the `re/emit_*.py` analog).
   *Blocked on the NCE descriptor layout — see the outer CLAUDE.md "long game".*
3. `npu_mem_alloc` NPU-visible buffers; stage `W`, `X` in, `Y` out.
4. `npu_graph_create` → init → exec (`VPU_CMD_INFERENCE_EXECUTE`), then read back `Y`.

Until steps 2–4 are real, flip nothing to "available": the CPU reference is the shipping path.
