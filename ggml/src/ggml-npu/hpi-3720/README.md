# hpi-3720 — Host-Platform Interface for Intel NPU 2.7 / VPU 3720

HPI exposes the C ABI used by `ggml-npu` for Q8_0 GEMM. The portable CPU
reference is always available. The optional Windows backend stages memory,
loads checked native blobs and submits Level Zero graphs. Missing or invalid
cache entries remain unavailable so the caller can use its CPU path.

## Files

| File | Responsibility |
|---|---|
| `hpi.h` | Public ABI, 34-byte Q8_0 block, portable FP16 conversions and row quantization |
| `hpi.c` | Shape validation, backend selection and portable CPU reference |
| `hpi_backend.h` | Private device structure and backend operation table |
| `hpi_npu_3720.c` | Device ownership, staging, graph records and execution |
| `hpi_npu_internal.h` | Pure repacking/residency and opt-in SIMD helpers, shared blob contract |
| `hpi_npu3720_slots.h` | Execution-slot ownership helpers, also checked with mock device types |
| `hpi_npu3720_blob.c` | Versioned disk-cache loading and identity validation |
| `hpi_gguf.h` / `hpi_gguf.c` | Public read-only GGUF reader |
| `tools/` | GGUF offload CLI and tiny Q8_0 fixture generator |
| `test/` | CPU GEMM, GGUF and pure expert-cache checks |

Pure data helpers share one internal header. Execution slots remain separate
because their tests supply mock device types. SIMD conversion is enabled in the internal header by
`HPI_NPU_INTERNAL_SIMD`, defined before its first include and after `npu.h`,
whose scalar functions define its rounding and NaN policy. This keeps its
exhaustive pure conversion test independent of the runtime. No compatibility include stubs are retained.

## Operation and ownership

```
W : N x K, Q8_0 (N contiguous rows of K/32 hpi_block_q8_0 blocks)
X : M x K, float32
Y : M x N, float32; Y[m,n] = sum_k dequant(W[n,k]) * X[m,k]
```

`K` must be divisible by `HPI_QK8_0` (32). Callers own the host buffers and
`hpi_device`; the backend owns its explicit private state and staging resources.
The CPU reference retains its per-block accumulation order. The public block
layout is guarded by a compile-time size assertion. Dispatcher validation,
status codes, function signatures and numerical bodies are unchanged by the
file consolidation.

## Build and test

From this directory:

```sh
cmake -S . -B build -DHPI3720_BUILD_TESTS=ON -DCMAKE_C_FLAGS=-Werror
cmake --build build
ctest --test-dir build --output-on-failure
./build/gguf_q8_0_offload model.gguf --list
./build/gguf_q8_0_offload model.gguf --tensor <name> --rows 32 --check
python tools/make_tiny_q8_0_gguf.py tiny-q8_0.gguf
```

The fixture generator needs NumPy and the checkout's `gguf-py`. Its random model
is only a path test. `test_gguf` also creates a small `hpi_synth_q8_0.gguf`.
The deterministic CPU GEMM test compares the same quantized weights against
independent double-precision accumulation; it does not exercise the NPU.

For integrated hardware builds, `GGML_NPU_HW=ON` adds the Windows runtime,
blob-loader translation unit, SHA256 source and required include paths.
`HPI_HAVE_NPU_3720` selects the runtime and `HPI_NPU3720_BLOB_READY` selects the
provider. `NPU_BLOB_CACHE` identifies the existing versioned cache at runtime.
The standalone `HPI3720_HAVE_NPU` option includes the runtime but deliberately
leaves the provider unavailable unless supplied by its caller. Override
`NPU_SRC_DIR`, `LEVEL_ZERO_INCLUDE` and `NPU_EXT_INCLUDE` for another layout.

Hardware progress and measured placement limits live in the outer repository's
`re/FINDINGS.md` and `re/PLAN.md`. File consolidation does not change routing or
establish a new hardware numerical/performance result.
