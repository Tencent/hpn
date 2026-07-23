/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 分阶段 AlltoAllv — Send/Recv 分离头文件
 *
 * 参考: DeepEP csrc/kernels/legacy/compiled.cuh: SEND_PHASE=1, RECV_PHASE=2
 *       DeepEP csrc/legacy/buffer.hpp: return_recv_hook 机制
 *       DeepEP tests/legacy/test_low_latency.py: 低延迟测试模式
 ************************************************************************/

#pragma once

#include <cstdint>
#include <vector>
#include <functional>

// 阶段位掩码 (对齐 DeepEP LEGACY_LOW_LATENCY_SEND_PHASE / RECV_PHASE)
enum AlltoAllPhase : uint32_t {
    SEND_PHASE = 1,   // 仅发送: 数据拷贝到send buffer + 提交网络传输
    RECV_PHASE = 2,   // 仅接收: 等待远端数据到达 + 数据拷贝到output
    FULL_PHASE = 3    // 发送+接收 (基线: 单kernel完成全部通信)
};

// 每个阶段的kernel配置
struct PhaseConfig {
    int num_sms;       // 该阶段使用的SM数
    int num_threads;   // 每个block的线程数
};

// AlltoAll 通信配置
struct AlltoAllConfig {
    int num_ranks;              // 总rank数
    int num_tokens_per_rank;    // 每个rank的token数
    int hidden_dim;             // 隐藏层维度
    int total_elements;         // num_tokens * hidden_dim

    PhaseConfig send_config;    // 发送阶段SM配置 (e.g. 24 SM, 256线程)
    PhaseConfig recv_config;    // 接收阶段SM配置 (e.g. 4 SM, 256线程)

    // 快慢卡模拟参数
    float slow_card_delay_ms;           // 慢卡的额外延迟(ms)
    int num_slow_ranks;                 // 慢卡数量
    std::vector<int> slow_rank_indices; // 哪些rank是慢卡

    AlltoAllConfig()
        : num_ranks(8), num_tokens_per_rank(128), hidden_dim(7168)
        , total_elements(0)
        , slow_card_delay_ms(2.0f), num_slow_ranks(2)
    {
        total_elements = num_tokens_per_rank * hidden_dim;
        send_config = {24, 256};  // 发送用满SM
        recv_config = {4, 256};   // 接收用少量SM，减少对计算kernel的抢占
    }
};

// 通信计时结果
struct AlltoAllResult {
    double send_time_ms;       // 发送阶段耗时
    double recv_time_ms;       // 接收阶段耗时
    double compute_time_ms;    // 重叠计算(GEMM)耗时
    double total_comm_time_ms; // 通信总耗时
    double end_to_end_ms;      // 端到端总耗时 (通信+计算)

    AlltoAllResult()
        : send_time_ms(0), recv_time_ms(0), compute_time_ms(0)
        , total_comm_time_ms(0), end_to_end_ms(0) {}
};

// 对比结果
struct ScenarioResult {
    AlltoAllResult phased_result;   // 分阶段方案结果
    AlltoAllResult baseline_result; // 基线方案结果
    double improvement_pct;         // 端到端改善百分比
};

/*
 * CUDA kernel: 根据 phases 位掩码执行发送、接收、或全部通信
 *
 * 模拟 AlltoAllv 通信模式:
 *   SEND: 从send_buf读数据，写入跨rank的交换区域（模拟RDMA发送）
 *   RECV: 从交换区域读回本rank的数据（模拟RDMA接收并写入recv_buf）
 *   FULL: 发送+接收依次完成
 *
 * slow_delay_factor: 慢卡的忙等待倍数（>1.0表示该rank有延迟）
 */
__global__ void alltoall_kernel(
    float* send_buf, float* recv_buf,
    int num_tokens, int hidden_dim,
    int rank, int num_ranks,
    float slow_delay_factor,
    uint32_t phases);

/*
 * 模拟重叠计算kernel (GEMM)
 * 对workspace执行FMA密集型循环，占用SM计算单元
 */
__global__ void gemm_overlap_kernel(
    float* workspace,
    int dim,
    int num_iterations);

/*
 * 分阶段AlltoAll: 先以SEND_PHASE启动kernel，返回recv_hook
 * 调用者在send和recv之间插入计算(GEMM)，然后调用recv_hook()完成接收
 *
 * 对齐 DeepEP buffer.hpp 的 return_recv_hook 机制:
 *   1. kernel以SEND_PHASE启动（仅发送）
 *   2. 返回recv_hook lambda
 *   3. 用户插入GEMM
 *   4. 调用recv_hook()以RECV_PHASE+更少SM启动kernel
 */
AlltoAllResult phased_alltoall(
    float* d_send_buf, float* d_recv_buf, float* d_gemm_workspace,
    const AlltoAllConfig& config,
    int rank,
    std::function<void()>& out_recv_hook);

/*
 * 基线AlltoAll: FULL_PHASE kernel一次性完成，然后顺序执行GEMM
 */
AlltoAllResult baseline_alltoall(
    float* d_send_buf, float* d_recv_buf, float* d_gemm_workspace,
    const AlltoAllConfig& config,
    int rank);

/*
 * 运行快慢卡对比测试
 * 在慢卡延迟场景下对比分阶段方案 vs 基线方案
 */
ScenarioResult run_fast_slow_scenario(
    const AlltoAllConfig& config,
    int num_repetitions);

// 校验通信正确性: 检查recv_buf已被正确写入
bool verify_correctness(const float* recv_buf, int total_elements, int rank);
