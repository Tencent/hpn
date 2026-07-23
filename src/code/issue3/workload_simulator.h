/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 工作负载模拟器 — 快慢卡场景生成与报告
 *
 * 模拟"快慢卡"现象: 部分rank因计算延迟晚到达AlltoAllv屏障，
 * 导致接收阶段长时间空等，浪费SM资源并抢占与之叠加的计算kernel
 ************************************************************************/

#pragma once

#include "phased_alltoall.h"

#include <vector>

// 生成测试数据（随机初始化发送buffer）
void generate_test_data(float* send_buf, int total_elements);

// 打印分阶段 vs 基线的格式化对比报告
void print_comparison_report(const ScenarioResult& result);
