# ggml-npu — Intel NPU 2.7 / VPU 3720 backend (work in progress)

## Normal launch on the measured Arrow Lake host

From the llama.cpp directory:

```powershell
python ggml/src/ggml-npu/run.py
python ggml/src/ggml-npu/run.py atomic
```

The default is Unsloth Flash-Next UD-Q4_K_XL on NPU with MTP 5. Atomic uses CPU
with MTP 3 because it won the paired coding measurements. `atomic --backend npu`
selects its NPU comparison. Both presets use four CPU threads, batch and microbatch
128, a CPU draft head, and CPU routed experts. Models are read directly from the
existing Hugging Face snapshots; the launcher checks every shard and the shared head.
The server listens on localhost:8080 with a 2048-token context and thinking disabled.

The launcher uses `build-npu-defaults/bin/llama-server.exe`; build from the parent NPU
workspace with `./src/build_llama.ps1 -Toolchain ucrt64 -BuildDir ./llama.cpp/build-npu-defaults`.
`--server` selects another hardware-enabled build, `--port` changes the port, and
`--print-only` displays the complete command without loading weights.

Whole-layer scheduling is now the backend default, including ordinary llama.cpp
invocations. Four internal CPU threads, turbo queues, F16C staging when supported,
an 8 GiB resident weight cache, and registered buffer/command-list reuse are also
defaults. Qwen4exp small F32 matrix reductions use the measured F32-dot policy.
No tuning environment variables are needed for these settings. `NPU_BLOB_CACHE`
still identifies external compiled weights/programs; the launcher chooses the
model's existing cache automatically. Direct invocations must supply a suitable cache
and keep routed experts on CPU (`-ncmoe 48` for these models) to avoid committing the
whole expert store to NPU host buffers.

Diagnostic overrides remain available: `NPU_OFFLOAD_GPU=0` selects legacy ACCEL
scheduling and `LLAMA_QWEN4EXP_F32_DOT=0` disables the small-batch F32 policy.
CPU/DPU overlap, early QKVZ scheduling and experimental expert dispatch remain off.
Low-bit expert SHAVE unpacking is not yet a production default; unsupported operations
stay on CPU. Never run a Level Zero metric streamer alongside the turbo queue.

These are coding presets selected from the 2026-09-06 matrix in the parent workspace,
`re/dive/perf_matrix_20260906/README.md`. CPU/NPU output identity remains unresolved.
Those performance measurements predate the host-allocation alignment fix; rebuilding
does not establish a new throughput result.

A ggml backend for the Intel NPU ("AI Boost", NPU 2.7 / VPU 37xx, PCI `8086:AD1D`) that talks to
the device **directly** — no OpenVINO runtime — reusing the direct-access toolkit developed in the
[`agentdavo/npu`](https://github.com/agentdavo/npu) project (`src/npu.h`: Level Zero → JSM/DMA →
the NCE/DPU MAC array).

Structurally this mirrors `ggml-hexagon`: a host-side backend (`ggml-npu.cpp`) plus
a device/interface layer. Here that layer is [`hpi-3720/`](hpi-3720/) — the **Host-Platform Interface
for VPU 3720**, the seam through which the host offloads compute to the accelerator.

## Status

| Piece | State |
|-------|-------|
| `hpi-3720/` HPI + Q8_0 GEMM contract | **done** — API + portable CPU reference, self-tested |
| `hpi-3720/` GGUF reader + offload harness | **done** — reads Q8_0 tensors from a `.gguf`, runs them through the offload |
| `ggml-npu.cpp` host backend glue | **done** — registers via `ggml_add_backend(NPU)`; llama enumerates it as a GPU-class device for whole-layer scheduling |
| `hpi-3720/` NPU-3720 hardware path | **implemented for eligible cached Q8_0 matrices** — Level Zero DPU programs with CPU fallback |

First target op is **Q8_0 offload** (a mul_mat whose weight operand is Q8_0-quantized), because Q8_0
is a clean, well-understood block format and a natural fit for the NPU's INT8 MAC array.

## Build & verify

```sh
cmake -S . -B build -DGGML_NPU=ON        # from the llama.cpp root
cmake --build build --target test-backend-ops -j
./build/bin/test-backend-ops test -b NPU -o MUL_MAT
```

This command builds the portable CPU-reference configuration. For real device execution use
`-DGGML_NPU_HW=ON` with the NPU toolkit headers, or the PowerShell build above.
Hardware dispatch requires a compatible compiled blob cache and eligible matrix geometry;
unsupported operations run on the internal ggml CPU backend.

The device is named `hpi-3720` (registration family `NPU`) and uses GPU-class
whole-layer placement by default. `--device hpi-3720` selects it. Keep routed
experts on CPU with `-ncmoe`/`-cmoe`; experimental expert dispatch is disabled.

### Prove it computes (no download)

Synthesize a tiny valid Q8_0 llama model, then run it and watch the backend log its work:

```sh
python3 ggml/src/ggml-npu/hpi-3720/tools/make_tiny_q8_0_gguf.py /tmp/tiny-q8_0.gguf   # needs numpy
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


### Device defaults (measured 2026-09-06 on AtomicChat/Flash-Next decode)

* **Turbo queue** (`ze_command_queue_desc_npu_ext_t.turbo`): ON by default; the small decode graphs are
  clock-bound and run 2.6-2.9x faster, whole-model DPU wait 206 -> 126 ms/token. `GGML_NPU_TURBO=0`
  opts out. Never open a Level Zero metric streamer while a turbo queue exists (recorded PC freeze).
* **F16C staging** of X/Y: ON by default when the CPU has F16C; `GGML_NPU_SIMD=0` selects scalar.
* **Weight-image budget** for v3 caches: 8 GiB by default (`GGML_NPU_WEIGHT_CACHE_MIB`), enough to keep a
  whole decode model's Q8_0 images resident; 512 MiB evicted every token.
* **Cache generation** is read from `NPU_BLOB_CACHE` (`cache-v3.ready`): a v3 cache holds two-tile
  weight-input programs with pre-swizzled slabs (both DPUs, both DMA engines, DPU CMX reads 2.7x faster
  than linear slabs) and admits the K=6144 and N=640 shapes a single tile cannot fit. Eligible ops are
  M=1 (1-row program) and M in 2..8 (8-row program, rows zero-padded): the latter is what makes the
  MTP verification batch run on the DPU. `re/build_blob_cache.py` writes both by default.
* `GGML_NPU_SPLIT_PCT` (CPU/NPU row split of large GEMVs) exists and is OFF: measured net negative,
  the two engines share ~40 GB/s of DDR on the target machine.

### Optional host overlap and SIMD staging

`GGML_NPU_OVERLAP=1` enables conservative CPU/NPU overlap in whole-layer
`NPU_OFFLOAD_GPU=1` mode. Independent CPU nodes following a queued DPU batch
run after its submission and before its completion wait. Tensor byte ranges
include aliases and strided spans; dependent or side-effecting nodes force a
flush. HPI remains synchronous to its caller, and slots are released only after
completion and output conversion. Unsupported/cache-missing operations still
use the existing CPU delegation. Default: disabled.

`GGML_NPU_SIMD` selects the runtime-checked AVX/F16C staging on GCC-compatible
x86 builds (default ON where F16C is present; `=0` for scalar). It converts eight
FP32/FP16 arguments at a time, preserving scalar rounding and NaN bits with scalar
tails. Only the conversion functions carry ISA target attributes; the generic driver
and portable reference retain their original build target. No weight format changes.

With `GGML_NPU_PROFILE=1`, `host_work_calls` counts callbacks actually run between
submission and wait, and `host_work_ms` records their CPU duration. That duration
is already included in `cpu_ms`; do not add it again. `overlap_candidates` counts
eligible CPU nodes, including candidates paired with cache misses. These are host
scopes, not hardware-engine overlap counters. Smaller remaining wait time alone
does not establish faster inference; compare total wall time and output correctness.

### Xe-LPG Level Zero shadow replay

`GGML_NPU_XE_LPG` compiles a disabled-by-default Xe-LPG replay path in a
`GGML_NPU_HW` build. Runtime activation also requires both of these variables:

```text
GGML_NPU_XE_LPG_SHADOW=1
GGML_NPU_XE_LPG_MODULE=<exact native Level Zero module>
```

`GGML_NPU_XE_LPG_CACHE_MIB` selects the packed expert cache budget (512 MiB by
default). `GGML_NPU_XE_LPG_SHADOW_BLOCK` selects one layer (block 0 by default).
`GGML_NPU_XE_LPG_PROFILE_EACH_GRAPH=1` prints a cumulative bridge and executor
profile after each successful graph compute. It is intended for correctness
gates whose long-lived server process cannot provide a destructor profile.
`GGML_NPU_XE_LPG_FUSED_LIST=1` selects an experimental one-command-list replay
with explicit activation and Q8_1 memory-range barriers. It defaults OFF; the
bridge passes the parsed mode into the executor explicitly.
`GGML_NPU_XE_LPG_OUTPUT_HASH=1` additionally hashes and logs each published
100 KiB output for diagnostic evidence. It defaults OFF to avoid a SHA-256 pass
and console I/O in timed execution. Completion, copying and copy verification
remain mandatory in either mode.
The cache key includes model and epoch identity, all three tensor identities,
layer, storage types, layout, expert ID, and matrix dimensions. Active route
slots cannot be evicted.

The current bounded gate accepts only the Flash-Next layout used by the local
validation model: Q4_K gate/up `[2560,640,512]`, Q5_1 down
`[640,2560,512]`, ten selected experts, and one to four token rows. It reads
expert data with the actual ggml 3-D tensor stride. The host quantizes each
input row to Q8_K, then Level Zero replays gate, up, exact SiLU, Q8_1
quantization, and down. It compares every down float bit-for-bit with the CPU
result.

Shadow mode retains CPU output. A separate `GGML_NPU_XE_LPG_REPLACE=1` opt-in
can replace the exact four-node M1 gate/up/SwiGLU/down island after a 1500 MiB
full-cache prefill and exact CPU/GPU canary. It pins the native module digest,
waits for completion into private GPU storage, then copies and verifies the
published output. Failure falls back to the held CPU island and disarms the
candidate. Grouped M2--M4 evaluation intentionally uses CPU. This bridge does
not advertise general asynchronous operations through `supports_op` or change
the backend's synchronous scheduler contract. The native module is
driver-specific and must be supplied explicitly. Defaults remain OFF until
complete execution beats the matched CPU/NPU control.
