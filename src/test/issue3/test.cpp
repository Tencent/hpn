/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: Test Function For Issue 3 — Phased AlltoAllv Validation
 *
 * Build:  cd ../code/issue3 && make test
 * Run:    ./test_runner
 ************************************************************************/

#include "phased_alltoall.h"
#include "workload_simulator.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { printf("  TEST: %s ... ", name); } while(0)

#define PASS() \
    do { printf("PASSED\n"); ++tests_passed; } while(0)

#define FAIL(msg) \
    do { printf("FAILED: %s\n", msg); ++tests_failed; } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while(0)

// --- Test: Phase enum values match DeepEP convention ---
static void test_phase_enum() {
    TEST("Phase enum values match DeepEP convention");
    ASSERT_TRUE(SEND_PHASE == 1, "SEND_PHASE should be 1");
    ASSERT_TRUE(RECV_PHASE == 2, "RECV_PHASE should be 2");
    ASSERT_TRUE(FULL_PHASE == 3, "FULL_PHASE should be 3");
    PASS();
}

// --- Test: GPU kernel launch and completion ---
static void test_kernel_basic() {
    TEST("AlltoAll kernel basic launch (FULL_PHASE)");
    int total = 128 * 7168;
    float *d_send, *d_recv;
    CUDA_CHECK(cudaMalloc(&d_send, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv, 8 * total * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_send, 0, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_recv, 0, 8 * total * sizeof(float)));

    alltoall_kernel<<<48, 256>>>(d_send, d_recv, 128, 7168, 0, 8, 1.0f, FULL_PHASE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // No crash = success
    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv));
    PASS();
}

// --- Test: SEND_PHASE + RECV_PHASE == FULL_PHASE (correctness) ---
static void test_phased_equivalence() {
    TEST("SEND + RECV phased equals FULL_PHASE");
    int tokens = 64;
    int hidden = 256;
    int total = tokens * hidden;

    float *d_send, *d_recv_full, *d_recv_phased;
    CUDA_CHECK(cudaMalloc(&d_send, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv_full, 8 * total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv_phased, 8 * total * sizeof(float)));

    // Init
    std::vector<float> h_send(total);
    for (int i = 0; i < total; ++i) h_send[i] = (float)(i + 1);
    CUDA_CHECK(cudaMemcpy(d_send, h_send.data(), total * sizeof(float), cudaMemcpyHostToDevice));

    // Full phase
    CUDA_CHECK(cudaMemset(d_recv_full, 0, 8 * total * sizeof(float)));
    alltoall_kernel<<<48, 256>>>(d_send, d_recv_full, tokens, hidden, 0, 8, 1.0f, FULL_PHASE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Phased: send then recv
    CUDA_CHECK(cudaMemset(d_recv_phased, 0, 8 * total * sizeof(float)));
    alltoall_kernel<<<48, 256>>>(d_send, d_recv_phased, tokens, hidden, 0, 8, 1.0f, SEND_PHASE);
    CUDA_CHECK(cudaDeviceSynchronize());
    alltoall_kernel<<<8, 256>>>(d_send, d_recv_phased, tokens, hidden, 0, 8, 1.0f, RECV_PHASE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compare
    std::vector<float> h_full(8 * total);
    std::vector<float> h_phased(8 * total);
    CUDA_CHECK(cudaMemcpy(h_full.data(), d_recv_full, 8 * total * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_phased.data(), d_recv_phased, 8 * total * sizeof(float), cudaMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < 8 * total; ++i) {
        if (std::fabs(h_full[i] - h_phased[i]) > 0.001f) {
            match = false;
            break;
        }
    }
    ASSERT_TRUE(match, "Phased result should match full result");

    CUDA_CHECK(cudaFree(d_send));
    CUDA_CHECK(cudaFree(d_recv_full));
    CUDA_CHECK(cudaFree(d_recv_phased));
    PASS();
}

// --- Test: Fast/slow card scenario produces valid results ---
static void test_fast_slow_scenario() {
    TEST("Fast/slow card scenario produces valid results");
    AlltoAllConfig config;
    config.num_ranks = 4;
    config.num_tokens_per_rank = 64;
    config.hidden_dim = 512;
    config.total_elements = config.num_tokens_per_rank * config.hidden_dim;
    config.send_config = {24, 256};
    config.recv_config = {4, 256};
    config.num_slow_ranks = 1;
    config.slow_rank_indices = {3};
    config.slow_card_delay_ms = 1.0f;

    ScenarioResult result = run_fast_slow_scenario(config, 5);

    ASSERT_TRUE(result.phased_result.end_to_end_ms > 0, "Phased E2E should be positive");
    ASSERT_TRUE(result.baseline_result.end_to_end_ms > 0, "Baseline E2E should be positive");
    ASSERT_TRUE(result.phased_result.send_time_ms > 0, "Send time should be positive");
    ASSERT_TRUE(result.phased_result.recv_time_ms > 0, "Recv time should be positive");

    printf("[phased=%.3f ms, baseline=%.3f ms, improvement=%.1f%%] ... ",
           result.phased_result.end_to_end_ms,
           result.baseline_result.end_to_end_ms,
           result.improvement_pct);
    PASS();
}

// --- Test: GEMM overlap kernel functional ---
static void test_gemm_kernel() {
    TEST("GEMM overlap kernel functional");
    int dim = 256;
    int total = dim * dim;
    float *d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_workspace, 0, total * sizeof(float)));

    gemm_overlap_kernel<<<24, 256>>>(d_workspace, dim, 5);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_ws(total);
    CUDA_CHECK(cudaMemcpy(h_ws.data(), d_workspace, total * sizeof(float), cudaMemcpyDeviceToHost));

    bool modified = false;
    for (int i = 0; i < total; ++i) {
        if (h_ws[i] != 0.0f) { modified = true; break; }
    }
    ASSERT_TRUE(modified, "GEMM kernel should modify workspace data");

    CUDA_CHECK(cudaFree(d_workspace));
    PASS();
}

// --- Test: Config defaults are reasonable ---
static void test_config_defaults() {
    TEST("Default config values are reasonable");
    AlltoAllConfig config;
    ASSERT_TRUE(config.num_ranks > 0, "num_ranks > 0");
    ASSERT_TRUE(config.num_tokens_per_rank > 0, "num_tokens > 0");
    ASSERT_TRUE(config.hidden_dim > 0, "hidden_dim > 0");
    ASSERT_TRUE(config.total_elements > 0, "total_elements computed");
    ASSERT_TRUE(config.send_config.num_sms > 0, "send_sms > 0");
    ASSERT_TRUE(config.recv_config.num_sms > 0, "recv_sms > 0");
    ASSERT_TRUE(config.recv_config.num_sms < config.send_config.num_sms,
                "recv_sms < send_sms (key optimization)");
    PASS();
}

int main() {
    printf("=== Issue 3: Phased AlltoAllv Tests ===\n\n");

    test_phase_enum();
    test_kernel_basic();
    test_phased_equivalence();
    test_fast_slow_scenario();
    test_gemm_kernel();
    test_config_defaults();

    printf("\n=== Results: %d passed, %d failed ===\n",
           tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
