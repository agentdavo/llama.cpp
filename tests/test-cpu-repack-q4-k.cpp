#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

ggml_backend_buffer_type_t ggml_backend_cpu_repack_buffer_type(void);

namespace {

[[noreturn]] void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        fail(message);
    }
}

struct graph_runner {
    ggml_context *        weight_ctx = nullptr;
    ggml_context *        graph_ctx  = nullptr;
    ggml_backend_buffer_t weight_buf = nullptr;
    ggml_threadpool_t     pool       = nullptr;
    ggml_tensor *         weight     = nullptr;
    ggml_tensor *         input      = nullptr;
    ggml_tensor *         output     = nullptr;
    ggml_cgraph *         graph      = nullptr;
    ggml_cplan            plan       = {};
    std::vector<uint8_t>  work;

    graph_runner(const std::vector<block_q4_K> & packed, const std::vector<float> & values,
                 int64_t k, int64_t rows, bool repacked, int threads) {
        require(k > 0 && k % QK_K == 0 && rows > 0 && rows % 8 == 0, "invalid matrix shape");
        require(packed.size() == static_cast<size_t>(rows * k / QK_K), "invalid packed size");
        require(values.size() == static_cast<size_t>(k), "invalid input size");

        weight_ctx = ggml_init({ 1024u * 1024u, nullptr, repacked });
        require(weight_ctx != nullptr, "weight context");
        weight = ggml_new_tensor_2d(weight_ctx, GGML_TYPE_Q4_K, k, rows);

        if (repacked) {
            weight_buf = ggml_backend_alloc_ctx_tensors_from_buft(weight_ctx, ggml_backend_cpu_repack_buffer_type());
            require(weight_buf != nullptr, "repack weight buffer");
            ggml_backend_tensor_set(weight, packed.data(), 0, packed.size() * sizeof(block_q4_K));
        } else {
            std::memcpy(weight->data, packed.data(), packed.size() * sizeof(block_q4_K));
        }

        graph_ctx = ggml_init({ 8u * 1024u * 1024u, nullptr, false });
        require(graph_ctx != nullptr, "graph context");
        input = ggml_new_tensor_2d(graph_ctx, GGML_TYPE_F32, k, 1);
        std::memcpy(input->data, values.data(), values.size() * sizeof(float));
        output = ggml_mul_mat(graph_ctx, weight, input);
        graph = ggml_new_graph_custom(graph_ctx, 16, false);
        ggml_build_forward_expand(graph, output);

        ggml_threadpool_params params = ggml_threadpool_params_default(threads);
        pool = ggml_threadpool_new(&params);
        require(pool != nullptr, "thread pool");
        plan = ggml_graph_plan(graph, threads, pool);
        work.resize(plan.work_size);
        plan.work_data = work.data();
    }

    ~graph_runner() {
        if (pool != nullptr) {
            ggml_threadpool_free(pool);
        }
        if (graph_ctx != nullptr) {
            ggml_free(graph_ctx);
        }
        if (weight_buf != nullptr) {
            ggml_backend_buffer_free(weight_buf);
        }
        if (weight_ctx != nullptr) {
            ggml_free(weight_ctx);
        }
    }

    void compute() {
        require(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS, "graph compute");
    }

    const float * data() const {
        return static_cast<const float *>(output->data);
    }
};

uint32_t rng_state = 0x9e3779b9u;
uint64_t output_hash = UINT64_C(1469598103934665603);

float random_float() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int32_t>(rng_state >> 8) - 0x00800000) * (1.0f / 2097152.0f);
}

void check_case(int64_t k, int64_t rows, int pattern) {
    std::vector<float> source(static_cast<size_t>(k * rows));
    std::vector<float> input(static_cast<size_t>(k));

    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < k; ++col) {
            float value;
            if (pattern == 0) {
                value = random_float();
            } else if (pattern == 1) {
                value = static_cast<float>((col + 3 * row) % 31) * (1.0f / 30.0f);
            } else {
                value = ((col + row) & 1) != 0 ? 8.0f : -7.0f;
            }
            source[static_cast<size_t>(row * k + col)] = value;
        }
    }
    for (int64_t col = 0; col < k; ++col) {
        input[static_cast<size_t>(col)] = pattern == 1 ? 1.0f : (pattern == 2 ? ((col & 1) ? 64.0f : -64.0f) : random_float());
    }

    std::vector<block_q4_K> packed(static_cast<size_t>(rows * k / QK_K));
    for (int64_t row = 0; row < rows; ++row) {
        quantize_row_q4_K_ref(source.data() + row * k, packed.data() + row * k / QK_K, k);
    }

    graph_runner reference(packed, input, k, rows, false, 1);
    graph_runner candidate(packed, input, k, rows, true, 1);
    reference.compute();
    candidate.compute();

    size_t changed = 0;
    float max_abs = 0.0f;
    float max_ref = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
        const float a = reference.data()[row];
        const float b = candidate.data()[row];
        require(std::isfinite(a) && std::isfinite(b), "non-finite result");
        if (std::memcmp(&a, &b, sizeof(float)) != 0) {
            ++changed;
        }
        max_abs = std::max(max_abs, std::fabs(a - b));
        max_ref = std::max(max_ref, std::fabs(a));
        const unsigned char * bytes = reinterpret_cast<const unsigned char *>(&b);
        for (size_t byte = 0; byte < sizeof(float); ++byte) {
            output_hash = (output_hash ^ bytes[byte]) * UINT64_C(1099511628211);
        }
    }
    const float tolerance = 5.0e-5f + 2.0e-5f * max_ref;
    if (max_abs > tolerance) {
        std::fprintf(stderr, "K=%lld rows=%lld pattern=%d changed=%zu max_abs=%g max_ref=%g tolerance=%g\n",
                     static_cast<long long>(k), static_cast<long long>(rows), pattern, changed, max_abs, max_ref, tolerance);
        fail("repacked Q4_K output exceeds the ordinary Q4_K tolerance");
    }
}

} // namespace

int main() {
    for (const int64_t k : { 256, 512, 2560 }) {
        for (const int64_t rows : { 8, 16, 64 }) {
            for (int pattern = 0; pattern < 3; ++pattern) {
                check_case(k, rows, pattern);
            }
        }
    }
    std::printf("PASS: 27 Q4_K repack cases match the ordinary kernel tolerance; digest=%016llx\n",
                static_cast<unsigned long long>(output_hash));
    return 0;
}
