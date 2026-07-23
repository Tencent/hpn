/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 带宽模型头文件 — SM扫描与带宽估算
 ************************************************************************/

#pragma once

#include "sm_budget_model.h"

#include <vector>
#include <string>

/*
 * 估算给定SM数时的通信带宽（归一化，1.0=链路饱和）
 *
 * 模型: 实际带宽 = min(SM处理能力, 链路容量)
 *   SM处理能力 = num_sms × 每SM带宽 / 流量需求（受限于读或写）
 *   链路容量   = 瓶颈带宽 / 瓶颈流量
 */
double estimateBandwidthAtSM(const EPConfig& config, const BandwidthParams& bw, int num_sms);

/*
 * 生成SM扫描数据 (从4到max_sms, 步长2)
 * 标记模型推荐的SM数为 is_recommended
 */
std::vector<SweepDataPoint> generateSMSweep(const EPConfig& config, const BandwidthParams& bw, int max_sms);

/*
 * 找到最优SM数: 边际带宽增益低于5%的拐点
 * 即比峰值带宽低5%时对应的SM数
 */
int findOptimalSM(const std::vector<SweepDataPoint>& sweep);

// 导出扫描数据为CSV格式
std::string exportSweepCSV(const std::vector<SweepDataPoint>& sweep, int recommended_sm);
