/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: Test Function For Issue 1 — SM/QP Budget Model Validation
 *
 * Build:  cd ../code/issue1 && make test
 * Run:    ./test_runner
 ************************************************************************/

#include "sm_budget_model.h"
#include "bandwidth_model.h"

#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { \
        std::cout << "  TEST: " << name << " ... "; \
    } while(0)

#define PASS() \
    do { \
        std::cout << "PASSED\n"; \
        ++tests_passed; \
    } while(0)

#define FAIL(msg) \
    do { \
        std::cout << "FAILED: " << msg << "\n"; \
        ++tests_failed; \
    } while(0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            FAIL(std::to_string(a) + " != " + std::to_string(b)); \
            return; \
        } \
    } while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { \
        if (std::fabs((a) - (b)) > (tol)) { \
            FAIL(std::to_string(a) + " not near " + std::to_string(b) \
                 + " (diff=" + std::to_string(std::fabs((a)-(b))) + ")"); \
            return; \
        } \
    } while(0)

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            FAIL("expected true"); \
            return; \
        } \
    } while(0)

// --- getExpectedTopK tests ---

static void test_expected_topk_basic() {
    TEST("getExpectedTopK basic");
    // 8 experts, topk=2, 2 groups
    // experts_per_group = 4, remaining = 4
    // C(4,2)/C(8,2) = 6/28 = 0.2143
    // expected = 2 * (1 - 0.2143) = 1.5714
    double val = getExpectedTopK(8, 2, 2);
    ASSERT_NEAR(val, 1.5714, 0.001);
    PASS();
}

static void test_expected_topk_single_group() {
    TEST("getExpectedTopK single group returns 0");
    double val = getExpectedTopK(8, 2, 1);
    ASSERT_EQ(val, 0.0);
    PASS();
}

static void test_expected_topk_all_groups() {
    TEST("getExpectedTopK with 8 groups (one expert per group)");
    // 8 experts, topk=2, 8 groups
    // experts_per_group = 1, remaining = 7
    // C(7,2)/C(8,2) = 21/28 = 0.75
    // expected = 8 * (1 - 0.75) = 2.0
    double val = getExpectedTopK(8, 2, 8);
    ASSERT_NEAR(val, 2.0, 0.001);
    PASS();
}

// --- computeSMBudget tests ---

static void test_sm_budget_default() {
    TEST("computeSMBudget with default config");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;
    bw.rdma_gbs     = 50;
    bw.nvlink_gbs   = 450;
    bw.sm_read_gbs  = 200;
    bw.sm_write_gbs = 50;

    SMBudgetResult r = computeSMBudget(config, bw);

    // Model should recommend fewer than 24 SMs
    ASSERT_TRUE(r.num_sms > 0);
    ASSERT_TRUE(r.num_sms < 24);
    // Should be even
    ASSERT_EQ(r.num_sms % 2, 0);
    // Should be at least 4
    ASSERT_TRUE(r.num_sms >= 4);
    // QP should be positive
    ASSERT_TRUE(r.num_qps > 0);

    std::cout << "  [result: num_sms=" << r.num_sms
              << ", num_qps=" << r.num_qps
              << ", bottleneck=" << (r.rdma_bottleneck ? "RDMA" : "NVLink")
              << "] ... ";
    PASS();
}

static void test_sm_budget_no_overlap() {
    TEST("computeSMBudget with prefer_overlap=false");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = false;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;
    SMBudgetResult r = computeSMBudget(config, bw);

    // Without overlap preference, SM count should be >= 64
    ASSERT_TRUE(r.num_sms >= 64);
    PASS();
}

static void test_sm_budget_single_node() {
    TEST("computeSMBudget single node (no scaleout)");
    EPConfig config;
    config.num_experts        = 256;
    config.num_topk           = 6;
    config.num_scaleout_ranks = 1;
    config.num_scaleup_ranks  = 8;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 128;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;
    bw.nvlink_gbs = 450;

    SMBudgetResult r = computeSMBudget(config, bw);

    ASSERT_TRUE(r.num_sms > 0);
    ASSERT_EQ(r.num_sms % 2, 0);
    ASSERT_TRUE(r.num_sms <= config.num_device_sms);
    PASS();
}

// --- QP budget tests ---

static void test_qp_direct_mode() {
    TEST("computeQPBudget direct mode");
    ASSERT_EQ(computeQPBudget(4, false, 256), 4);   // min(4, 9)=4
    ASSERT_EQ(computeQPBudget(10, false, 256), 9);  // min(10, 9)=9
    ASSERT_EQ(computeQPBudget(8, false, 256), 8);   // min(8, 9)=8
    ASSERT_EQ(computeQPBudget(16, false, 256), 9);  // min(16, 9)=9
    PASS();
}

static void test_qp_hybrid_mode() {
    TEST("computeQPBudget hybrid mode");
    ASSERT_EQ(computeQPBudget(4, true, 256), 65);   // 4*16+1=65
    ASSERT_EQ(computeQPBudget(8, true, 256), 129);  // 8*16+1=129
    ASSERT_EQ(computeQPBudget(16, true, 256), 256); // 16*16+1=257 capped to 256
    PASS();
}

// --- Bandwidth estimate tests ---

static void test_bandwidth_monotonic() {
    TEST("Bandwidth monotonically increases with SM count");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;

    double prev = 0.0;
    for (int sm = 4; sm <= 132; sm += 2) {
        double bw_val = estimateBandwidthAtSM(config, bw, sm);
        ASSERT_TRUE(bw_val >= prev - 1e-9);  // non-decreasing
        prev = bw_val;
    }
    PASS();
}

static void test_bandwidth_95_percent() {
    TEST("Model-recommended SM achieves >= 95% of peak bandwidth");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;

    auto sweep = generateSMSweep(config, bw, 132);
    SMBudgetResult r = computeSMBudget(config, bw);

    double peak_bw = sweep.back().bandwidth;
    double model_bw = estimateBandwidthAtSM(config, bw, r.num_sms);
    double ratio = model_bw / peak_bw;

    std::cout << "  [model_bw=" << model_bw << ", peak_bw=" << peak_bw
              << ", ratio=" << (ratio * 100.0) << "%] ... ";
    ASSERT_TRUE(ratio >= 0.95);
    PASS();
}

static void test_sweep_recommended_marked() {
    TEST("Sweep marks exactly one SM as recommended");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;

    auto sweep = generateSMSweep(config, bw, 132);
    int count = 0;
    for (const auto& pt : sweep) {
        if (pt.is_recommended) ++count;
    }
    ASSERT_EQ(count, 1);
    PASS();
}

static void test_sm_savings_vs_24() {
    TEST("Model SM < 24 (SM savings positive)");
    EPConfig config;
    config.num_experts        = 288;
    config.num_topk           = 8;
    config.num_scaleout_ranks = 8;
    config.num_scaleup_ranks  = 1;
    config.num_nvlink_ranks   = 8;
    config.num_device_sms     = 132;
    config.prefer_overlap     = true;
    config.num_allocated_qps  = 256;
    config.allow_hybrid_mode  = false;

    BandwidthParams bw;
    SMBudgetResult r = computeSMBudget(config, bw);

    int sm_saved = 24 - r.num_sms;

    std::cout << "  [saved " << sm_saved << " SMs vs fixed 24] ... ";
    ASSERT_TRUE(sm_saved > 0);
    PASS();
}

int main() {
    std::cout << "=== Issue 1: SM/QP Budget Model Tests ===\n\n";

    // Combinatorics
    test_expected_topk_basic();
    test_expected_topk_single_group();
    test_expected_topk_all_groups();

    // SM budget
    test_sm_budget_default();
    test_sm_budget_no_overlap();
    test_sm_budget_single_node();

    // QP budget
    test_qp_direct_mode();
    test_qp_hybrid_mode();

    // Bandwidth
    test_bandwidth_monotonic();
    test_bandwidth_95_percent();
    test_sweep_recommended_marked();
    test_sm_savings_vs_24();

    std::cout << "\n=== Results: " << tests_passed << " passed, "
              << tests_failed << " failed ===\n";

    return tests_failed > 0 ? 1 : 0;
}
