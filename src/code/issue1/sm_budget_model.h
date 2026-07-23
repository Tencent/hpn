/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 分析式 SM/QP 预算模型头文件
 *
 * 参考: DeepEP deep_ep/buffers/elastic.py
 *       get_theoretical_num_sms() (行 729-853)
 *       get_theoretical_num_qps() (行 836-853)
 ************************************************************************/

#pragma once

#include <string>
#include <vector>

// EP 配置参数
struct EPConfig {
    int num_experts;           // 总专家数 (e.g. 288)
    int num_topk;              // 每token top-k选择 (e.g. 8)
    int num_scaleout_ranks;    // 跨节点rank数 (RDMA)
    int num_scaleup_ranks;     // 节点内rank数 (NVLink)
    int num_nvlink_ranks;      // NVLink可达rank数
    int num_device_sms;        // 设备SM总数 (e.g. 132 for H100)
    bool prefer_overlap;       // 是否优先计算-通信叠加
    int num_allocated_qps;     // 最大可用QP数
    bool allow_hybrid_mode;    // 是否允许hybrid模式

    int totalRanks() const { return num_scaleout_ranks * num_scaleup_ranks; }
};

// 带宽参数 (GB/s)
struct BandwidthParams {
    double rdma_gbs      = 0;   // 0=自动 (使用默认值 50 GB/s)
    double nvlink_gbs    = 0;   // 0=自动 (使用默认值 450 GB/s)
    double sm_read_gbs   = 200; // 每个SM的HBM读带宽
    double sm_write_gbs  = 50;  // 每个SM的HBM写带宽
};

// 模型输出结果
struct SMBudgetResult {
    int num_sms;               // 推荐SM数
    int num_qps;               // 推荐QP数
    bool rdma_bottleneck;      // 是否RDMA为瓶颈

    // 中间过程值（用于分析）
    double num_expected_topk;           // 期望topk选择数（全部rank）
    double num_expected_scaleout_topk;  // 期望topk选择数（跨节点rank）
    double sm_read_traffic;             // HBM读流量
    double sm_write_traffic;            // HBM写流量
    double rdma_traffic;                // RDMA流量
    double nvlink_traffic;              // NVLink流量
    double bounded_traffic;             // 瓶颈链路流量
    double bounded_gbs;                 // 瓶颈链路带宽
    double raw_num_sms;                 // 安全系数/对齐前的原始SM数
};

// SM扫描数据点
struct SweepDataPoint {
    int num_sms;               // SM数量
    double bandwidth;          // 该SM数下的带宽（归一化，1.0=链路饱和）
    bool is_recommended;       // 是否为模型推荐的SM数
};

/*
 * 计算期望topk选择数
 *   公式: num_groups * (1 - C(experts - experts/groups, topk) / C(experts, topk))
 *   含义: 随机topk选择中，至少有一项落在组外（需要跨组通信）的期望token数
 *   当 num_groups == 1 时返回0（无需跨组通信）
 */
double getExpectedTopK(int num_experts, int num_topk, int num_groups);

/*
 * 核心: 根据EP配置和带宽参数计算SM预算
 * 完整复现 DeepEP elastic.py 中 get_theoretical_num_sms() 的算法:
 *   1. 计算期望topk
 *   2. 按通信阶段累加 HBM读/写、RDMA、NVLink 流量
 *   3. 比较 rdma_traffic/rdma_gbs vs nvlink_traffic/nvlink_gbs 识别瓶颈
 *   4. 反推所需SM数: max(瓶颈*读/读带宽, 瓶颈*写/写带宽)
 *   5. 后处理: ×1.25安全系数 → 偶数对齐 → 下限4(非overlap时64) → cap到设备上限
 */
SMBudgetResult computeSMBudget(const EPConfig& config, const BandwidthParams& bw);

/*
 * 计算QP预算
 *   direct模式: min(num_sms, 8+1)
 *   hybrid模式: num_sms * 16 + 1
 *   结果不超过 num_allocated_qps
 */
int computeQPBudget(int num_sms, bool allow_hybrid, int num_allocated_qps);

// 向上对齐
inline int alignUp(int value, int alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}
