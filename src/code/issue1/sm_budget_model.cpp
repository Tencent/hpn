/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 分析式 SM/QP 预算模型实现
 *
 * 参考: DeepEP deep_ep/buffers/elastic.py
 *       get_theoretical_num_sms() (行 729-853)
 *       get_theoretical_num_qps() (行 836-853)
 ************************************************************************/

#include "sm_budget_model.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

/*
 * 用 lgamma 计算组合数对数，避免大数溢出
 * C(n, k) = n! / (k! * (n-k)!)
 * log C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)
 */
static double logComb(int n, int k) {
    if (k < 0 || k > n) return -std::numeric_limits<double>::infinity();
    if (k == 0 || k == n) return 0.0;
    return std::lgamma(n + 1) - std::lgamma(k + 1) - std::lgamma(n - k + 1);
}

// C(n, k) = exp(logComb(n, k))，下溢时返回0
static double comb(int n, int k) {
    double lc = logComb(n, k);
    if (lc < -700) return 0.0;  // 低于双精度范围
    return std::exp(lc);
}

double getExpectedTopK(int num_experts, int num_topk, int num_groups) {
    if (num_groups <= 1) return 0.0;
    int experts_per_group = num_experts / num_groups;
    int remaining = num_experts - experts_per_group;
    // 所有topk都落在本组内的概率
    double p_all_local = comb(remaining, num_topk) / comb(num_experts, num_topk);
    // 至少有一个落在组外（需要通信）的概率
    double p_any_remote = 1.0 - p_all_local;
    return num_groups * p_any_remote;
}

// 默认带宽值: RDMA 50 GB/s (CX7 400Gbps每rank), NVLink 450 GB/s (H100每rank)
static double defaultRDMA_GBS()  { return 50.0; }
static double defaultNVLinkGBS() { return 450.0; }

SMBudgetResult computeSMBudget(const EPConfig& config, const BandwidthParams& bw) {
    int total_ranks  = config.totalRanks();
    int num_scaleout = config.num_scaleout_ranks;
    int num_scaleup  = config.num_scaleup_ranks;
    int num_nvlink   = config.num_nvlink_ranks;

    double rdma_gbs   = (bw.rdma_gbs   > 0) ? bw.rdma_gbs   : defaultRDMA_GBS();
    double nvlink_gbs = (bw.nvlink_gbs > 0) ? bw.nvlink_gbs : defaultNVLinkGBS();

    // 步骤1: 计算期望topk值
    double num_expected_topk = getExpectedTopK(config.num_experts, config.num_topk, total_ranks);
    double num_expected_scaleout_topk = (num_scaleout > 1)
        ? getExpectedTopK(config.num_experts, config.num_topk, num_scaleout) : 0.0;

    // 步骤2: 按通信阶段累加流量
    double sm_read  = 0.0;
    double sm_write = 0.0;
    double rdma_traffic  = 0.0;
    double nvlink_traffic = 0.0;

    if (num_scaleout > 1) {
        // scale-out (hybrid) 模式: 跨节点通信
        sm_read  += 1.0 / num_expected_topk;  // 从HBM读token
        sm_write += 1.0 / num_expected_topk;  // 写RDMA send buffer (scaleup warp)

        // 本地旁路写
        sm_write += (1.0 / num_expected_topk) * (num_expected_scaleout_topk / num_scaleout);

        // 跨节点RDMA流量
        rdma_traffic += (1.0 / num_expected_topk)
            * (num_expected_scaleout_topk * (1.0 - 1.0 / num_scaleout));

        // forward warp: 读数据并发起节点内通信
        sm_read  += num_expected_scaleout_topk / num_expected_topk;
        sm_write += 1.0;

        // NVLink流量
        nvlink_traffic += 1.0 - (1.0 / num_scaleup);
    } else {
        // direct (非scale-out) 模式: 纯节点内通信
        int num_rdma_ranks = total_ranks - num_nvlink;
        if (num_rdma_ranks > 1) {
            sm_write += 1.0 / num_expected_topk;  // 写send buffer
        }

        if (total_ranks > 1) {
            sm_write += (double)num_nvlink / total_ranks;  // 发起NVLink
        }

        // NVLink流量（排除本地旁路）
        if (num_nvlink > 1 && total_ranks > 1) {
            nvlink_traffic += ((double)num_nvlink / total_ranks) * (1.0 - 1.0 / num_nvlink);
        }

        // RDMA流量
        if (total_ranks > 1) {
            rdma_traffic += (double)(total_ranks - num_nvlink) / total_ranks;
        }
    }

    // 步骤3: 识别瓶颈链路
    // 比较 rdma_traffic/rdma_gbs 和 nvlink_traffic/nvlink_gbs，值更大的是瓶颈
    bool rdma_bottleneck = false;
    double bounded_traffic, bounded_gbs;

    if (num_scaleout > 1) {
        double rdma_ratio   = (rdma_gbs   > 0) ? rdma_traffic  / rdma_gbs   : 0.0;
        double nvlink_ratio = (nvlink_gbs > 0) ? nvlink_traffic / nvlink_gbs : 0.0;
        if (rdma_ratio > nvlink_ratio) {
            bounded_traffic = rdma_traffic;
            bounded_gbs     = rdma_gbs;
            rdma_bottleneck = true;
        } else {
            bounded_traffic = nvlink_traffic;
            bounded_gbs     = nvlink_gbs;
        }
    } else {
        bounded_traffic = nvlink_traffic;
        bounded_gbs     = nvlink_gbs;
        if (rdma_traffic > nvlink_traffic && rdma_gbs > 0) {
            bounded_traffic = rdma_traffic;
            bounded_gbs     = rdma_gbs;
            rdma_bottleneck = true;
        }
    }

    // 步骤4: 反推所需SM数
    // SM数 = max(瓶颈带宽/瓶颈流量 × HBM读需求/HBM读带宽, 瓶颈带宽/瓶颈流量 × HBM写需求/HBM写带宽)
    double raw_num_sms;
    if (bounded_traffic <= 0.0 || bounded_gbs <= 0.0) {
        raw_num_sms = config.num_device_sms;  // 无流量时返回全部SM（如EP=1）
    } else {
        double sm_read_req  = (bounded_gbs / bounded_traffic) * (sm_read  / bw.sm_read_gbs);
        double sm_write_req = (bounded_gbs / bounded_traffic) * (sm_write / bw.sm_write_gbs);
        raw_num_sms = std::max(sm_read_req, sm_write_req);
    }

    // 步骤5: 后处理
    // ceil → ×1.25安全系数 → 偶数对齐 → 下限4 → cap到设备SM数
    int num_sms = alignUp(std::max(4, (int)std::ceil(raw_num_sms * 1.25)), 2);

    if (!config.prefer_overlap) {
        num_sms = std::max(num_sms, 64);  // 不叠加计算时至少64 SM
    }
    num_sms = std::min(num_sms, config.num_device_sms);  // 不超过设备SM总数

    int num_qps = computeQPBudget(num_sms, config.allow_hybrid_mode, config.num_allocated_qps);

    SMBudgetResult result;
    result.num_sms                   = num_sms;
    result.num_qps                   = num_qps;
    result.rdma_bottleneck           = rdma_bottleneck;
    result.num_expected_topk         = num_expected_topk;
    result.num_expected_scaleout_topk = num_expected_scaleout_topk;
    result.sm_read_traffic           = sm_read;
    result.sm_write_traffic          = sm_write;
    result.rdma_traffic              = rdma_traffic;
    result.nvlink_traffic            = nvlink_traffic;
    result.bounded_traffic           = bounded_traffic;
    result.bounded_gbs               = bounded_gbs;
    result.raw_num_sms               = raw_num_sms;

    return result;
}

int computeQPBudget(int num_sms, bool allow_hybrid, int num_allocated_qps) {
    int num_qps;
    if (allow_hybrid) {
        // hybrid模式: 每个channel和notify各占一个独立QP
        num_qps = num_sms * 16 + 1;
    } else {
        // direct模式: 减少QP以降低DB ring开销
        num_qps = std::min(num_sms, 8 + 1);
    }
    return std::min(num_qps, num_allocated_qps);
}
