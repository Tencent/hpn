/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 工作负载模拟器 — 快慢卡数据生成与报告输出
 ************************************************************************/

#include "workload_simulator.h"

#include <cstdio>
#include <cmath>
#include <algorithm>
#include <random>

void generate_test_data(float* send_buf, int total_elements) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < total_elements; ++i) {
        send_buf[i] = dist(rng);
    }
}

void print_comparison_report(const ScenarioResult& result) {
    auto& p = result.phased_result;
    auto& b = result.baseline_result;

    printf("\n");
    printf("  +============================================================+\n");
    printf("  |  AlltoAllv Send/Recv 分阶段叠加 — 对比报告                  |\n");
    printf("  +============================================================+\n");
    printf("  | %-24s | %10s | %10s |\n", "指标", "基线", "分阶段");
    printf("  +--------------------------+------------+------------+\n");
    printf("  | %-24s | %8.3f ms | %8.3f ms |\n",
           "发送时间", b.send_time_ms, p.send_time_ms);
    printf("  | %-24s | %8.3f ms | %8.3f ms |\n",
           "接收时间", b.recv_time_ms, p.recv_time_ms);
    printf("  | %-24s | %8.3f ms | %8.3f ms |\n",
           "GEMM计算时间", b.compute_time_ms, p.compute_time_ms);
    printf("  +--------------------------+------------+------------+\n");
    printf("  | %-24s | %8.3f ms | %8.3f ms |\n",
           "端到端总时间", b.end_to_end_ms, p.end_to_end_ms);
    printf("  +--------------------------+------------+------------+\n");
    printf("\n");

    if (result.improvement_pct > 0) {
        printf("  [改善] 分阶段方案端到端时间降低 %.2f%%\n", result.improvement_pct);
    } else if (result.improvement_pct < 0) {
        printf("  [退化] 分阶段方案端到端时间增加 %.2f%%\n", -result.improvement_pct);
    } else {
        printf("  [持平] 无明显差异\n");
    }

    printf("\n  分析:\n");
    printf("    基线:    FULL_PHASE kernel (全24 SM) → 顺序GEMM\n"
           "            → GEMM在通信完成后执行，被接收等待阻塞\n");
    printf("    分阶段:  SEND_PHASE kernel → GEMM (与接收重叠) → RECV_PHASE (4 SM)\n"
           "            → GEMM与接收并行，接收只占少量SM，计算获更多SM资源\n");
    printf("    SM节省:  接收使用4 SM vs 基线24 SM → "
           "重叠期间释放20 SM给计算kernel\n");
    printf("\n");
}
