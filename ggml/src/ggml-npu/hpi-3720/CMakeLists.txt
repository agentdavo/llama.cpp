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
    hpi_npu_3720.c
    hpi_gguf.c)

target_include_directories(hpi3720 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (HPI3720_HAVE_NPU)
    # hpi_npu_3720.c owns the src/npu.h implementation and needs the full Level Zero layer. Default
    # the include paths to the repo's own copies (this dir lives five levels below <npu>); override
    # on the cmake line for an out-of-tree checkout.
    set(NPU_SRC_DIR       "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src"
        CACHE PATH "Path to the src/ direct-access toolkit (contains npu.h)")
    set(LEVEL_ZERO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../re/ref/level-zero/include"
        CACHE PATH "Path to the Level Zero API headers")
    set(NPU_EXT_INCLUDE   "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../re/ref/npu-ext"
        CACHE PATH "Path to the NPU (ze_graph) extension headers")

    target_compile_definitions(hpi3720 PRIVATE HPI_HAVE_NPU_3720 NPU_NO_GDN)
    target_include_directories(hpi3720 PRIVATE ${NPU_SRC_DIR} ${LEVEL_ZERO_INCLUDE} ${NPU_EXT_INCLUDE})
    # Define HPI_NPU3720_BLOB_READY and add the blob-seam TU once the compute path exists on the box.
endif()

# Warnings-as-errors on the pure code (outer CLAUDE.md rule 1), where the compiler supports it.
if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hpi3720 PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Wcast-align -Wpointer-arith)
endif()

# gguf_q8_0_offload: point the offload at a real .gguf model (or the synthetic one from the tests).
add_executable(gguf_q8_0_offload gguf_q8_0_offload.c)
target_link_libraries(gguf_q8_0_offload PRIVATE hpi3720)
if (NOT WIN32)
    target_link_libraries(gguf_q8_0_offload PRIVATE m)
endif()

if (HPI3720_BUILD_TESTS)
    enable_testing()
    foreach (t test_q8_0_gemm test_gguf)
        add_executable(${t} test/${t}.c)
        target_link_libraries(${t} PRIVATE hpi3720)
        if (NOT WIN32)
            target_link_libraries(${t} PRIVATE m)
        endif()
        add_test(NAME hpi3720_${t} COMMAND ${t})
    endforeach()
endif()
