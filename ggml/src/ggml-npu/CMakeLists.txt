# ggml-npu — Intel NPU 2.7 / VPU 3720 backend.
#
# The backend glue (ggml-npu.cpp) plus the cross-platform hpi-3720 Host-Platform Interface compiled
# straight in. The portable CPU reference has no NPU dependency; the optional
# HPI_HAVE_NPU_3720 path supplies the Windows Level Zero runtime and blob loader.

ggml_add_backend_library(ggml-npu
                         ggml-npu.cpp
                         hpi-3720/hpi.c
                         hpi-3720/hpi_npu_3720.c
                         hpi-3720/hpi_npu3720_blob.c
                        )

target_include_directories(ggml-npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

option(GGML_NPU_HW "ggml-npu: compile the real VPU-3720 hardware path (Windows x64 + NPU + src/ toolkit)" OFF)
option(GGML_NPU_XE_LPG "ggml-npu: compile the disabled-by-default Xe-LPG Level Zero shadow replay" OFF)
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
    target_sources(ggml-npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/hash/sha256/sha256.c)
    target_include_directories(ggml-npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor/hash)

endif()

if (GGML_NPU_XE_LPG)
    if (NOT GGML_NPU_HW)
        message(FATAL_ERROR "GGML_NPU_XE_LPG requires GGML_NPU_HW")
    endif()
    target_compile_definitions(ggml-npu PRIVATE GGML_NPU_XE_LPG)
    target_sources(ggml-npu PRIVATE ggml-npu-xe-lpg.cpp xe-lpg/xe_lpg_executor.c)
endif()
