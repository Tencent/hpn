/*************************************************************************
 * Copyright (c) 2025, TENCENT CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * Author: moningchen@tencent.com
 * Content: Main Function For Issue 1 — Analytical SM/QP Budget Model
 *
 * Build:  make
 * Usage:
 *   ./sm_budget_model --mode single  [EP params...]
 *   ./sm_budget_model --mode sweep   [EP params...]
 *   ./sm_budget_model --mode compare [EP params...]
 ************************************************************************/

#include "sm_budget_model.h"
#include "bandwidth_model.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cmath>

static void printSeparator() {
    std::cout << std::string(72, '-') << std::endl;
}

static void modeSingle(const EPConfig& config, const BandwidthParams& bw) {
    SMBudgetResult r = computeSMBudget(config, bw);

    std::cout << std::fixed << std::setprecision(4);
    printSeparator();
    std::cout << "  Analytical SM/QP Budget Model — Single Config\n";
    printSeparator();
    std::cout << "INPUT:\n";
    std::cout << "  num_experts        = " << config.num_experts << "\n";
    std::cout << "  num_topk           = " << config.num_topk << "\n";
    std::cout << "  num_scaleout_ranks = " << config.num_scaleout_ranks << "\n";
    std::cout << "  num_scaleup_ranks  = " << config.num_scaleup_ranks << "\n";
    std::cout << "  num_nvlink_ranks   = " << config.num_nvlink_ranks << "\n";
    std::cout << "  num_device_sms     = " << config.num_device_sms << "\n";
    std::cout << "  prefer_overlap     = " << (config.prefer_overlap ? "true" : "false") << "\n";
    std::cout << "  rdma_gbs           = " << bw.rdma_gbs << " (0=auto)\n";
    std::cout << "  nvlink_gbs         = " << bw.nvlink_gbs << " (0=auto)\n";
    std::cout << "  sm_read_gbs        = " << bw.sm_read_gbs << "\n";
    std::cout << "  sm_write_gbs       = " << bw.sm_write_gbs << "\n";
    std::cout << "\nINTERMEDIATE VALUES:\n";
    std::cout << "  num_expected_topk          = " << r.num_expected_topk << "\n";
    std::cout << "  num_expected_scaleout_topk = " << r.num_expected_scaleout_topk << "\n";
    std::cout << "  sm_read_traffic            = " << r.sm_read_traffic << "\n";
    std::cout << "  sm_write_traffic           = " << r.sm_write_traffic << "\n";
    std::cout << "  rdma_traffic               = " << r.rdma_traffic << "\n";
    std::cout << "  nvlink_traffic             = " << r.nvlink_traffic << "\n";
    std::cout << "  bounded_traffic            = " << r.bounded_traffic
              << " (" << (r.rdma_bottleneck ? "RDMA" : "NVLink") << " bottleneck)\n";
    std::cout << "  bounded_gbs                = " << r.bounded_gbs << "\n";
    std::cout << "  raw_num_sms (pre-process)  = " << r.raw_num_sms << "\n";
    std::cout << "\nOUTPUT:\n";
    std::cout << "  Recommended SM count = " << r.num_sms << "\n";
    std::cout << "  Recommended QP count = " << r.num_qps << "\n";
    printSeparator();
}

static void modeSweep(const EPConfig& config, const BandwidthParams& bw) {
    int max_sms = config.num_device_sms;
    auto sweep = generateSMSweep(config, bw, max_sms);
    SMBudgetResult r = computeSMBudget(config, bw);
    int optimal_sm = findOptimalSM(sweep);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "# SM vs Bandwidth Sweep\n";
    std::cout << "# Model recommended SM: " << r.num_sms << "\n";
    std::cout << "# 95%-saturation SM:    " << optimal_sm << "\n";
    std::cout << "# Peak bandwidth:       " << sweep.back().bandwidth << "\n";
    std::cout << "# Bandwidth at model SM: "
              << estimateBandwidthAtSM(config, bw, r.num_sms) << "\n";
    std::cout << "#\n";
    std::cout << exportSweepCSV(sweep, r.num_sms);
}

static void modeCompare(const EPConfig& config, const BandwidthParams& bw) {
    const int FIXED_SM = 24;
    SMBudgetResult r = computeSMBudget(config, bw);

    double bw_model    = estimateBandwidthAtSM(config, bw, r.num_sms);
    double bw_fixed_24 = estimateBandwidthAtSM(config, bw, FIXED_SM);

    int sm_saved = FIXED_SM - r.num_sms;
    double sm_saved_pct = 100.0 * sm_saved / FIXED_SM;

    std::cout << std::fixed << std::setprecision(2);
    printSeparator();
    std::cout << "  SM Savings Analysis: Model vs Fixed 24 SM\n";
    printSeparator();
    std::cout << std::setprecision(4);
    std::cout << "  Model recommended SM:  " << r.num_sms << "\n";
    std::cout << "  Fixed baseline SM:     " << FIXED_SM << "\n";
    std::cout << "  SM saved:              " << sm_saved
              << " (" << sm_saved_pct << "%)\n\n";
    std::cout << "  Bandwidth at model SM: " << bw_model
              << " (" << (bw_model * 100.0) << "% of link capacity)\n";
    std::cout << "  Bandwidth at 24 SM:    " << bw_fixed_24
              << " (" << (bw_fixed_24 * 100.0) << "% of link capacity)\n";
    std::cout << "  Model BW / Fixed BW:   " << (bw_model / bw_fixed_24 * 100.0) << "%\n\n";

    if (bw_model >= 0.95) {
        std::cout << "  [PASS] Model bandwidth >= 95% of link capacity.\n";
    } else {
        std::cout << "  [WARN] Model bandwidth < 95% of link capacity.\n";
    }

    if (sm_saved > 0) {
        std::cout << "  [INFO] Model saves " << sm_saved << " SMs vs fixed 24 SM,"
                  << " freeing resources for GEMM computation.\n";
        std::cout << "  End-to-end benefit: " << sm_saved << " SMs x "
                  << bw.sm_read_gbs << " GB/s read + " << bw.sm_write_gbs
                  << " GB/s write available for compute overlap.\n";
    } else if (sm_saved < 0) {
        std::cout << "  [INFO] Model recommends MORE SMs than 24, ensuring bandwidth is not compromised.\n";
    } else {
        std::cout << "  [INFO] Model recommends same SM count as baseline (24).\n";
    }
    printSeparator();
}

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --mode <single|sweep|compare>   Operation mode (default: single)\n"
              << "  --num-experts <int>             Total experts (default: 288)\n"
              << "  --num-topk <int>                Top-k per token (default: 8)\n"
              << "  --num-scaleout-ranks <int>      Cross-node ranks (default: 8)\n"
              << "  --num-scaleup-ranks <int>       Intra-node ranks (default: 1)\n"
              << "  --num-nvlink-ranks <int>        NVLink ranks (default: 8)\n"
              << "  --num-device-sms <int>          Device SM count (default: 132)\n"
              << "  --rdma-gbs <float>              RDMA bandwidth GB/s (default: 0=auto)\n"
              << "  --nvlink-gbs <float>            NVLink bandwidth GB/s (default: 0=auto)\n"
              << "  --sm-read-gbs <float>           Per-SM HBM read GB/s (default: 200)\n"
              << "  --sm-write-gbs <float>          Per-SM HBM write GB/s (default: 50)\n"
              << "  --prefer-overlap <0|1>          Prefer compute-comm overlap (default: 1)\n"
              << "  --allow-hybrid <0|1>            Allow hybrid QP mode (default: 0)\n"
              << "  --num-allocated-qps <int>       Allocated QP count (default: 256)\n"
              << "  --help                          Show this message\n";
}

int main(int argc, char* argv[]) {
    // Default config: 288 experts, topk=8, 8 scaleout ranks, single scaleup, H100-like
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
    bw.rdma_gbs     = 0;    // auto
    bw.nvlink_gbs   = 0;    // auto
    bw.sm_read_gbs  = 200;
    bw.sm_write_gbs = 50;

    std::string mode = "single";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--num-experts" && i + 1 < argc) {
            config.num_experts = std::stoi(argv[++i]);
        } else if (arg == "--num-topk" && i + 1 < argc) {
            config.num_topk = std::stoi(argv[++i]);
        } else if (arg == "--num-scaleout-ranks" && i + 1 < argc) {
            config.num_scaleout_ranks = std::stoi(argv[++i]);
        } else if (arg == "--num-scaleup-ranks" && i + 1 < argc) {
            config.num_scaleup_ranks = std::stoi(argv[++i]);
        } else if (arg == "--num-nvlink-ranks" && i + 1 < argc) {
            config.num_nvlink_ranks = std::stoi(argv[++i]);
        } else if (arg == "--num-device-sms" && i + 1 < argc) {
            config.num_device_sms = std::stoi(argv[++i]);
        } else if (arg == "--rdma-gbs" && i + 1 < argc) {
            bw.rdma_gbs = std::stod(argv[++i]);
        } else if (arg == "--nvlink-gbs" && i + 1 < argc) {
            bw.nvlink_gbs = std::stod(argv[++i]);
        } else if (arg == "--sm-read-gbs" && i + 1 < argc) {
            bw.sm_read_gbs = std::stod(argv[++i]);
        } else if (arg == "--sm-write-gbs" && i + 1 < argc) {
            bw.sm_write_gbs = std::stod(argv[++i]);
        } else if (arg == "--prefer-overlap" && i + 1 < argc) {
            config.prefer_overlap = std::stoi(argv[++i]) != 0;
        } else if (arg == "--allow-hybrid" && i + 1 < argc) {
            config.allow_hybrid_mode = std::stoi(argv[++i]) != 0;
        } else if (arg == "--num-allocated-qps" && i + 1 < argc) {
            config.num_allocated_qps = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate
    if (config.num_experts % config.totalRanks() != 0) {
        std::cerr << "Error: num_experts must be divisible by total ranks ("
                  << config.totalRanks() << ")\n";
        return 1;
    }
    if (config.num_scaleout_ranks > 1
        && config.num_experts % config.num_scaleout_ranks != 0) {
        std::cerr << "Error: num_experts must be divisible by num_scaleout_ranks ("
                  << config.num_scaleout_ranks << ")\n";
        return 1;
    }

    if (mode == "single") {
        modeSingle(config, bw);
    } else if (mode == "sweep") {
        modeSweep(config, bw);
    } else if (mode == "compare") {
        modeCompare(config, bw);
    } else {
        std::cerr << "Unknown mode: " << mode << ". Use single, sweep, or compare.\n";
        return 1;
    }

    return 0;
}
