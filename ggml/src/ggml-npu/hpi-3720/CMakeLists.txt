# ggml-npu / hpi-3720 — Host-Platform Interface for the Intel NPU 2.7 (VPU 3720).
#
# Standalone, cross-platform static library (C11, no ggml/CPU-feature dependencies) that provides the
# Q8_0 GEMM offload seam with a portable CPU reference. It is intentionally NOT yet wired into the
# ggml backend registry (ggml/src/CMakeLists.txt ggml_add_backend) so it cannot break the main build;
# add it there once the NPU_3720 path is real. See README.md.
#
# Build & test standalone:
#     cmake -S . -B build -DHPI3720_BUILD_TESTS=ON && cmake --build build && ctest --test-dir build

cmake_minimum_required(VERSION 3.14)
project(hpi3720 C)

option(HPI3720_BUILD_TESTS "hpi-3720: build the Q8_0 GEMM self-test" OFF)
option(HPI3720_HAVE_NPU    "hpi-3720: compile the real VPU-3720 hardware path (Windows x64 + NPU)" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(hpi3720 STATIC
    hpi.c
    hpi_cpu.c
    hpi_npu_3720.c)

target_include_directories(hpi3720 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (HPI3720_HAVE_NPU)
    target_compile_definitions(hpi3720 PRIVATE HPI_HAVE_NPU_3720)
    # The real path pulls in the src/ direct-access toolkit (npu.h, Level Zero). Wire those includes
    # here when implementing hpi_npu_3720.c, e.g.:
    #   target_include_directories(hpi3720 PRIVATE ${NPU_SRC_DIR} ${LEVEL_ZERO_INCLUDE})
    #   target_compile_definitions(hpi3720 PRIVATE NPU_NO_GDN)
endif()

# Warnings-as-errors on the pure code (outer CLAUDE.md rule 1), where the compiler supports it.
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hpi3720 PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wcast-align -Wpointer-arith)
endif()

if (HPI3720_BUILD_TESTS)
    enable_testing()
    add_executable(test_q8_0_gemm test/test_q8_0_gemm.c)
    target_link_libraries(test_q8_0_gemm PRIVATE hpi3720)
    if (NOT WIN32)
        target_link_libraries(test_q8_0_gemm PRIVATE m)
    endif()
    add_test(NAME hpi3720_q8_0_gemm COMMAND test_q8_0_gemm)
endif()
