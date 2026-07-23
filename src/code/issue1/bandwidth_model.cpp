/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: 带宽模型实现 — SM扫描与带宽估算
 ************************************************************************/

#include "bandwidth_model.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>

double estimateBandwidthAtSM(const EPConfig& config, const BandwidthParams& bw, int num_sms) {
    if (num_sms <= 0) return 0.0;

    SMBudgetResult result = computeSMBudget(config, bw);

    if (result.bounded_traffic <= 0.0 || result.bounded_gbs <= 0.0) {
        return 1.0;  // 无流量时带宽视为最大
    }

    // SM处理能力: 受限于读或写的较小值
    double sm_read_cap  = num_sms * bw.sm_read_gbs  / result.sm_read_traffic;
    double sm_write_cap = num_sms * bw.sm_write_gbs / result.sm_write_traffic;
    double sm_cap = std::min(sm_read_cap, sm_write_cap);

    // 链路瓶颈容量
    double link_cap = result.bounded_gbs / result.bounded_traffic;

    // 实际带宽 = min(SM能力, 链路容量)
    double effective = std::min(sm_cap, link_cap);

    // 归一化: 1.0 = 链路饱和
    double bw_normalized = effective / link_cap;
    return std::min(bw_normalized, 1.0);
}

std::vector<SweepDataPoint> generateSMSweep(const EPConfig& config, const BandwidthParams& bw, int max_sms) {
    std::vector<SweepDataPoint> sweep;
    SMBudgetResult result = computeSMBudget(config, bw);
    int recommended_sm = result.num_sms;

    for (int sm = 4; sm <= max_sms; sm += 2) {
        SweepDataPoint pt;
        pt.num_sms        = sm;
        pt.bandwidth      = estimateBandwidthAtSM(config, bw, sm);
        pt.is_recommended = (sm == recommended_sm);
        sweep.push_back(pt);
    }
    return sweep;
}

int findOptimalSM(const std::vector<SweepDataPoint>& sweep) {
    if (sweep.empty()) return 4;

    // 达到峰值带宽95%所需的SM数即视为最优
    double peak_bw = sweep.back().bandwidth;
    int optimal_sm = sweep[0].num_sms;

    for (size_t i = 0; i < sweep.size(); ++i) {
        if (sweep[i].bandwidth >= peak_bw * 0.95) {
            optimal_sm = sweep[i].num_sms;
            break;
        }
    }
    return optimal_sm;
}

std::string exportSweepCSV(const std::vector<SweepDataPoint>& sweep, int recommended_sm) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "num_sms,bandwidth,is_recommended\n";
    for (const auto& pt : sweep) {
        oss << pt.num_sms << ","
            << pt.bandwidth << ","
            << (pt.num_sms == recommended_sm ? "1" : "0") << "\n";
    }
    return oss.str();
}
