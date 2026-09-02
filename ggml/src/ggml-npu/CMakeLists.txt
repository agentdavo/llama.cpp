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
                        )

target_include_directories(ggml-npu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

option(GGML_NPU_HW "ggml-npu: compile the real VPU-3720 hardware path (Windows x64 + NPU + src/ toolkit)" OFF)
if (GGML_NPU_HW)
    target_compile_definitions(ggml-npu PRIVATE HPI_HAVE_NPU_3720)
    # When implementing hpi_npu_3720.c, wire the src/ direct-access toolkit here, e.g.:
    #   target_include_directories(ggml-npu PRIVATE ${NPU_SRC_DIR} ${LEVEL_ZERO_INCLUDE})
    #   target_compile_definitions(ggml-npu PRIVATE NPU_NO_GDN)
endif()
