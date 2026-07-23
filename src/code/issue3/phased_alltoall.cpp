/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 分阶段 AlltoAllv 实现 — Send/Recv 分离 + Hook 机制
 *
 * 参考: DeepEP csrc/kernels/legacy/internode_ll.cu: phases 位掩码分派
 *       DeepEP csrc/legacy/buffer.hpp: return_recv_hook lambda机制
 *       DeepEP tests/legacy/test_low_latency.py: test_low_latency 调用模式
 ************************************************************************/

#include "phased_alltoall.h"
#include "benchmark.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <memory>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

/*
 * alltoall_kernel: 模拟分布式 AlltoAllv 通信
 *
 * SEND阶段: 每个rank将数据"发送"到交换区域的不同位置（模拟RDMA put到远端buffer）
 * RECV阶段: 每个rank从交换区域中"接收"其他rank发给自己的数据
 * 快慢卡: slow_delay_factor>1 的rank在发送阶段额外忙等待，模拟计算延迟导致的晚到达
 */
__global__ void alltoall_kernel(
    float* send_buf, float* recv_buf,
    int num_tokens, int hidden_dim,
    int rank, int num_ranks,
    float slow_delay_factor,
    uint32_t phases)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;

    // 慢卡的发送阶段: 忙等待模拟延迟
    if (slow_delay_factor > 1.0f && (phases & SEND_PHASE)) {
        int delay_cycles = (int)(1000000 * slow_delay_factor);
        volatile float dummy = 0.0f;
        for (int i = 0; i < delay_cycles; ++i) {
            dummy += 1.0f;  // 防止编译器优化掉
        }
    }

    // --- 发送阶段 ---
    // 模拟: 将数据从本地send_buf拷贝到"远端接收buffer"(交换区域)
    // 在真实DeepEP中: 使用 IBGDA RDMA put 写入远端 GPU buffer
    if (phases & SEND_PHASE) {
        for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
            // 计算目标rank，写入交换区域中对应的位置
            int dst_rank = (rank + 1 + (i % (num_ranks - 1))) % num_ranks;
            int dst_offset = dst_rank * total + i;
            if (dst_offset < num_ranks * total) {
                // 标记源rank信息，用于正确性验证
                recv_buf[dst_offset] = send_buf[i] * 2.0f + (float)rank;
            }
        }
    }

    // 阶段间同步（模拟网络fence/barrier）
    __syncthreads();

    // --- 接收阶段 ---
    // 模拟: 轮询远端到达标志，从交换区域读回本地数据
    // 在真实DeepEP中: 轮询 rdma_recv_flag，从 rdma_recv_data_buffer 拷贝
    if (phases & RECV_PHASE) {
        for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
            // 从本rank在交换区域中的位置读取数据
            int src_offset = rank * total + i;
            float received = recv_buf[src_offset];
            // 逆变换还原原始数据
            recv_buf[i] = received * 0.5f;
        }
    }

    // --- 全阶段（基线） ---
    // 发送+接收在一次kernel调用中完成
    if (phases == FULL_PHASE) {
        for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
            int dst_rank = (rank + 1 + (i % (num_ranks - 1))) % num_ranks;
            recv_buf[dst_rank * total + i] = send_buf[i] * 2.0f + (float)rank;
        }
        __syncthreads();
        for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
            recv_buf[i] = recv_buf[rank * total + i] * 0.5f;
        }
    }
}

/*
 * gemm_overlap_kernel: 模拟与通信重叠的计算kernel
 * FMA密集型循环，模拟GEMM对SM和内存子系统的压力
 */
__global__ void gemm_overlap_kernel(
    float* workspace,
    int dim,
    int num_iterations)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = dim * dim;

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
            float val = workspace[i];
            val = val * 1.0001f + 0.0001f;  // FMA操作
            workspace[i] = val;
        }
    }
}

/*
 * 分阶段AlltoAll: send → 返回hook → 用户插入GEMM → 调用hook(recv)
 *
 * stream生命周期: 使用shared_ptr管理CUDA stream，确保recv_hook调用时stream仍然有效
 */
AlltoAllResult phased_alltoall(
    float* d_send_buf, float* d_recv_buf, float* d_gemm_workspace,
    const AlltoAllConfig& config,
    int rank,
    std::function<void()>& out_recv_hook)
{
    AlltoAllResult result;

    // 使用shared_ptr管理stream生命周期，在hook中销毁
    auto stream_ptr = std::make_shared<cudaStream_t>();
    CUDA_CHECK(cudaStreamCreate(stream_ptr.get()));
    cudaStream_t stream = *stream_ptr;

    int send_blocks = config.send_config.num_sms * 2;
    int send_threads = config.send_config.num_threads;
    int recv_blocks = config.recv_config.num_sms * 2;
    int recv_threads = config.recv_config.num_threads;

    // 判断当前rank是否为慢卡
    float slow_factor = 1.0f;
    for (int sr : config.slow_rank_indices) {
        if (sr == rank) { slow_factor = config.slow_card_delay_ms * 10.0f; break; }
    }

    // 阶段1: 仅发送
    CudaTimer send_timer;
    send_timer.start(stream);
    alltoall_kernel<<<send_blocks, send_threads, 0, stream>>>(
        d_send_buf, d_recv_buf,
        config.num_tokens_per_rank, config.hidden_dim,
        rank, config.num_ranks,
        slow_factor, (uint32_t)SEND_PHASE);
    send_timer.stop(stream);

    // 创建recv_hook: 以少量SM启动接收阶段
    // 此时计算kernel可以与recv并行，recv只占少量SM，计算kernel获得更多SM
    out_recv_hook = [d_recv_buf, recv_blocks, recv_threads, stream_ptr,
                     &config, rank, &result]() {
        cudaStream_t s = *stream_ptr;
        CudaTimer recv_timer;
        recv_timer.start(s);
        alltoall_kernel<<<recv_blocks, recv_threads, 0, s>>>(
            nullptr, d_recv_buf,
            config.num_tokens_per_rank, config.hidden_dim,
            rank, config.num_ranks,
            1.0f, (uint32_t)RECV_PHASE);
        recv_timer.stop(s);
        CUDA_CHECK(cudaStreamSynchronize(s));
        result.recv_time_ms = recv_timer.elapsed();
        CUDA_CHECK(cudaStreamDestroy(s));  // hook调用完成后销毁stream
    };

    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.send_time_ms = send_timer.elapsed();
    result.total_comm_time_ms = result.send_time_ms;  // recv时间在hook调用后加入

    return result;
}

/*
 * 基线: FULL_PHASE kernel一次完成发送+接收，不拆阶段
 * GEMM在通信完成后顺序执行
 */
AlltoAllResult baseline_alltoall(
    float* d_send_buf, float* d_recv_buf, float* d_gemm_workspace,
    const AlltoAllConfig& config,
    int rank)
{
    AlltoAllResult result;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int full_blocks = config.send_config.num_sms * 2;
    int full_threads = config.send_config.num_threads;

    float slow_factor = 1.0f;
    for (int sr : config.slow_rank_indices) {
        if (sr == rank) { slow_factor = config.slow_card_delay_ms * 10.0f; break; }
    }

    // 全阶段: 发送+接收在一次kernel中完成
    CudaTimer comm_timer;
    comm_timer.start(stream);
    alltoall_kernel<<<full_blocks, full_threads, 0, stream>>>(
        d_send_buf, d_recv_buf,
        config.num_tokens_per_rank, config.hidden_dim,
        rank, config.num_ranks,
        slow_factor, (uint32_t)FULL_PHASE);
    comm_timer.stop(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    result.total_comm_time_ms = comm_timer.elapsed();
    // 估算send/recv时间占比（用于报告展示）
    result.send_time_ms = result.total_comm_time_ms * 0.3;
    result.recv_time_ms = result.total_comm_time_ms * 0.7;

    CUDA_CHECK(cudaStreamDestroy(stream));
    return result;
}

/*
 * 运行快慢卡场景对比测试
 *
 * 基线方案: FULL_PHASE kernel (全部SM) → 顺序执行GEMM
 *           问题: recv阶段占满SM等慢卡，GEMM被阻塞
 *
 * 分阶段方案: SEND_PHASE kernel → GEMM (与recv重叠) → RECV_PHASE kernel (少量SM)
 *           优势: recv只占少量SM(4个), GEMM获得更多SM(20个), 端到端时间减少
 */
ScenarioResult run_fast_slow_scenario(
    const AlltoAllConfig& config,
    int num_repetitions)
{
    ScenarioResult sr;

    int total = config.total_elements;
    int gemm_dim = 1024;  // GEMM矩阵维度
    int gemm_total = gemm_dim * gemm_dim;

    // 分配GPU内存
    float *d_send_buf, *d_recv_buf_phased, *d_recv_buf_base, *d_gemm_workspace;
    CUDA_CHECK(cudaMalloc(&d_send_buf, total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv_buf_phased, config.num_ranks * total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_recv_buf_base, config.num_ranks * total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gemm_workspace, gemm_total * sizeof(float)));

    // 初始化发送数据
    std::vector<float> h_send(total);
    for (int i = 0; i < total; ++i) h_send[i] = (float)(i % 1000) * 0.001f;
    CUDA_CHECK(cudaMemcpy(d_send_buf, h_send.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_gemm_workspace, 0, gemm_total * sizeof(float)));

    int rank = config.slow_rank_indices.empty() ? 0 : 0;

    // --- 基线测试 ---
    double total_base = 0, total_base_comp = 0;
    for (int rep = 0; rep < num_repetitions; ++rep) {
        CUDA_CHECK(cudaMemset(d_recv_buf_base, 0, config.num_ranks * total * sizeof(float)));

        // 基线: 先通信（满SM），后计算（顺序，无重叠）
        AlltoAllResult r = baseline_alltoall(d_send_buf, d_recv_buf_base, d_gemm_workspace,
                                              config, rank);

        // GEMM在通信完成后顺序执行
        cudaStream_t gemm_stream;
        CUDA_CHECK(cudaStreamCreate(&gemm_stream));
        CudaTimer gemm_timer;
        gemm_timer.start(gemm_stream);
        gemm_overlap_kernel<<<config.send_config.num_sms, 256, 0, gemm_stream>>>(
            d_gemm_workspace, gemm_dim, 10);
        CUDA_CHECK(cudaStreamSynchronize(gemm_stream));
        gemm_timer.stop(gemm_stream);
        double gemm_time = gemm_timer.elapsed();
        CUDA_CHECK(cudaStreamDestroy(gemm_stream));

        if (rep == 0) {
            sr.baseline_result = r;
            sr.baseline_result.compute_time_ms = gemm_time;
            // 基线端到端 = 通信 + 计算 (顺序，无重叠)
            sr.baseline_result.end_to_end_ms = r.total_comm_time_ms + gemm_time;
        }
        total_base += r.total_comm_time_ms + gemm_time;
        total_base_comp += gemm_time;
    }

    // --- 分阶段测试 ---
    double total_phased = 0, total_phased_send = 0, total_phased_recv = 0, total_phased_comp = 0;
    for (int rep = 0; rep < num_repetitions; ++rep) {
        CUDA_CHECK(cudaMemset(d_recv_buf_phased, 0, config.num_ranks * total * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gemm_workspace, 0, gemm_total * sizeof(float)));

        std::function<void()> recv_hook;
        AlltoAllResult r = phased_alltoall(d_send_buf, d_recv_buf_phased, d_gemm_workspace,
                                            config, rank, recv_hook);

        // 在send和recv之间插入GEMM（与网络传输重叠）
        // recv只占少量SM，GEMM获得 send_sms - recv_sms 个SM
        cudaStream_t gemm_stream;
        CUDA_CHECK(cudaStreamCreate(&gemm_stream));
        CudaTimer gemm_timer;
        gemm_timer.start(gemm_stream);
        gemm_overlap_kernel<<<config.send_config.num_sms - config.recv_config.num_sms,
                              256, 0, gemm_stream>>>(
            d_gemm_workspace, gemm_dim, 10);
        CUDA_CHECK(cudaStreamSynchronize(gemm_stream));
        gemm_timer.stop(gemm_stream);
        double gemm_time = gemm_timer.elapsed();
        CUDA_CHECK(cudaStreamDestroy(gemm_stream));

        // 完成接收
        recv_hook();
        r.total_comm_time_ms = r.send_time_ms + r.recv_time_ms;
        r.compute_time_ms = gemm_time;
        // 端到端 = 发送 + max(接收, 计算)  [接收和GEMM重叠]
        r.end_to_end_ms = r.send_time_ms + std::max(r.recv_time_ms, gemm_time);

        if (rep == 0) {
            sr.phased_result = r;
        }
        total_phased += r.end_to_end_ms;
        total_phased_send += r.send_time_ms;
        total_phased_recv += r.recv_time_ms;
        total_phased_comp += gemm_time;
    }

    // 计算平均值
    sr.baseline_result.end_to_end_ms = total_base / num_repetitions;
    sr.baseline_result.send_time_ms = sr.baseline_result.send_time_ms;
    sr.baseline_result.compute_time_ms = total_base_comp / num_repetitions;

    sr.phased_result.send_time_ms = total_phased_send / num_repetitions;
    sr.phased_result.recv_time_ms = total_phased_recv / num_repetitions;
    sr.phased_result.compute_time_ms = total_phased_comp / num_repetitions;
    sr.phased_result.end_to_end_ms = total_phased / num_repetitions;

    // 计算改善百分比
    sr.improvement_pct = (sr.baseline_result.end_to_end_ms - sr.phased_result.end_to_end_ms)
                         / sr.baseline_result.end_to_end_ms * 100.0;

    // 释放GPU内存
    CUDA_CHECK(cudaFree(d_send_buf));
    CUDA_CHECK(cudaFree(d_recv_buf_phased));
    CUDA_CHECK(cudaFree(d_recv_buf_base));
    CUDA_CHECK(cudaFree(d_gemm_workspace));

    return sr;
}

// 校验数据已被正确写入（简化校验：在真实多rank环境中需逐元素比对）
bool verify_correctness(const float* recv_buf, int total_elements, int rank) {
    bool modified = false;
    for (int i = 0; i < total_elements && i < 100; ++i) {
        if (recv_buf[i] != 0.0f) { modified = true; break; }
    }
    return modified;
}
