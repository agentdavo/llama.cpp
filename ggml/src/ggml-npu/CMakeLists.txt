# ggml-npu — Intel NPU 2.7 / VPU 3720 backend.
#
# The backend glue (ggml-npu.cpp) plus the cross-platform hpi-3720 Host-Platform Interface compiled
# straight in (the CPU reference; the real VPU-3720 path is stubbed behind HPI_HAVE_NPU_3720). No NPU
# or OS-specific dependency, so it builds and enumerates as a device on any platform.

ggml_add_backend_library(ggml-npu
                         ggml-npu.cpp
                         hpi-3720/hpi.c
                         hpi-3720/hpi_cpu.c
                         hpi-3720/hpi_npu_3720.c
                         hpi-3720/hpi_npu3720_blob.c
                        )

target_include_directories(ggml-npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

option(GGML_NPU_HW "ggml-npu: compile the real VPU-3720 hardware path (Windows x64 + NPU + src/ toolkit)" OFF)
if (GGML_NPU_HW)
    # hpi_npu_3720.c owns the src/npu.h implementation (NPU_IMPLEMENTATION) and needs the full Level
    # Zero layer, so it requires the direct-access toolkit and the L0 / NPU-extension headers on the
    # include path. Point these at the repo's own copies (defaults assume the ggml tree lives under
    # <npu>/llama.cpp, i.e. src/ and re/ref/ are two levels up); override on the cmake line if not.
    set(NPU_SRC_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/../../../../src"
        CACHE PATH "Path to the src/ direct-access toolkit (contains npu.h)")
    set(LEVEL_ZERO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../re/ref/level-zero/include"
        CACHE PATH "Path to the Level Zero API headers")
    set(NPU_EXT_INCLUDE   "${CMAKE_CURRENT_SOURCE_DIR}/../../../../re/ref/npu-ext"
        CACHE PATH "Path to the NPU (ze_graph) extension headers")

    # HPI_HAVE_NPU_3720: the real Level-Zero device path (hpi_npu_3720.c). NPU_NO_GDN: hpi_npu_3720.c
    # drops the AVX GDN host math from npu.h. HPI_NPU3720_BLOB_READY: the compute seam is real — the
    # provider TU hpi_npu3720_blob.c loads per-tensor blobs precompiled by re/build_blob_cache.py
    # (set NPU_BLOB_CACHE at run time to the cache dir), so npu_available() advertises the device.
    target_compile_definitions(ggml-npu PRIVATE HPI_HAVE_NPU_3720 HPI_NPU3720_BLOB_READY NPU_NO_GDN)
    target_include_directories(ggml-npu PRIVATE ${NPU_SRC_DIR} ${LEVEL_ZERO_INCLUDE} ${NPU_EXT_INCLUDE})

    # The blob compute seam (hpi_npu3720_build_blob) is a separate TU implemented on the box; define
    # HPI_NPU3720_BLOB_READY once it exists so npu_available() advertises the device. Until then the
    # backend stays honest (reports unavailable, AUTO falls back to the CPU reference).
    #   target_compile_definitions(ggml-npu PRIVATE HPI_NPU3720_BLOB_READY)
    #   target_sources(ggml-npu PRIVATE ${NPU_BLOB_IMPL})
endif()
