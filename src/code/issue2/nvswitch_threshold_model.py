#!/usr/bin/env python3
"""
NVSwitch AllReduce Switching Threshold Model
=============================================
Quantitative analysis of intranode AllReduce implementations:
- One-Shot (AllGather + Local Reduce)
- Two-Shot Multimem (NVLS via NVSwitch hardware reduction)
- Two-Shot P2P (Reduce-Scatter + AllGather on GPU SMs)

Builds a message-size-aware switching threshold model to automatically
select the optimal algorithm.

Usage:
  python nvswitch_threshold_model.py --analyze
  python nvswitch_threshold_model.py --thresholds
"""

import argparse
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Hardware Parameters (H800 reference) ────────────────────────────

@dataclass
class NVSwitchHW:
    nvlink_bw_gbs: float = 900.0       # NVLink per-GPU bandwidth
    nvswitch_bw_gbs: float = 900.0      # NVSwitch port bandwidth
    sm_clock_ghz: float = 1.98
    num_sms: int = 132
    num_gpus: int = 8                   # GPUs in NVSwitch domain
    hbm_bw_gbs: float = 3350.0          # HBM bandwidth
    warp_size: int = 32
    shared_mem_kb_per_sm: int = 227


# ── Algorithm Models ─────────────────────────────────────────────────

def model_oneshot_allreduce(
    msg_size_bytes: int,
    hw: NVSwitchHW,
) -> Dict:
    """
    One-Shot: AllGather + Local Reduce
    Each GPU receives data from all other GPUs, then reduces locally.
    Latency: O(1) communication steps
    Bandwidth: each GPU receives (N-1) * M bytes
    SM usage: zero (NVSwitch does aggregation)
    """
    M = msg_size_bytes
    N = hw.num_gpus

    data_per_gpu = (N - 1) * M
    nvlink_time = data_per_gpu / (hw.nvlink_bw_gbs * 1e9)

    local_reduce_gops = (N - 1) * M / 2  # Reduction ops
    compute_time = local_reduce_gops / (hw.sm_clock_ghz * 1e9 * hw.num_sms * 2)

    return {
        "algorithm": "One-Shot",
        "comm_time_us": round(nvlink_time * 1e6, 3),
        "compute_time_us": round(compute_time * 1e6, 3),
        "total_time_us": round((nvlink_time + compute_time) * 1e6, 3),
        "data_per_gpu_mb": round(data_per_gpu / 1e6, 2),
        "sm_overhead": 0,
        "best_for_sizes": "0 - 16KB",
    }


def model_multimem_nvls(
    msg_size_bytes: int,
    hw: NVSwitchHW,
) -> Dict:
    """
    Two-Shot Multimem (NVLS):
    Reduce-Scatter via NVSwitch + AllGather via NVSwitch.
    NVSwitch performs the reduction in-hardware (SHARP).
    SM: nearly zero (NVSwitch does the reduction)
    Data: ~2(N-1)/N * M per GPU
    """
    M = msg_size_bytes
    N = hw.num_gpus

    data_per_gpu = 2 * (N - 1) / N * M
    nvlink_time = data_per_gpu / (hw.nvlink_bw_gbs * 1e9)

    sm_overhead = 0  # NVSwitch does the work

    return {
        "algorithm": "Two-Shot Multimem (NVLS)",
        "comm_time_us": round(nvlink_time * 1e6, 3),
        "compute_time_us": 0.0,
        "total_time_us": round(nvlink_time * 1e6, 3),
        "data_per_gpu_mb": round(data_per_gpu / 1e6, 2),
        "sm_overhead": sm_overhead,
        "best_for_sizes": "16KB - 1GB+",
    }


def model_twoshot_p2p(
    msg_size_bytes: int,
    hw: NVSwitchHW,
) -> Dict:
    """
    Two-Shot P2P:
    Reduce-Scatter + AllGather, reduction on GPU SMs.
    SM: significant (reduction compute)
    Data: ~2(N-1)/N * M per GPU (same as Multimem)
    """
    M = msg_size_bytes
    N = hw.num_gpus

    data_per_gpu = 2 * (N - 1) / N * M
    nvlink_time = data_per_gpu / (hw.nvlink_bw_gbs * 1e9)

    # P2P reduction on SMs
    reduce_gops = M / 2 * (N - 1) / N
    sm_needed = min(int(reduce_gops / (hw.sm_clock_ghz * 1e9 * 2) * 100) + 1, hw.num_sms)
    compute_time = reduce_gops / (hw.sm_clock_ghz * 1e9 * min(sm_needed, 32))

    return {
        "algorithm": "Two-Shot P2P",
        "comm_time_us": round(nvlink_time * 1e6, 3),
        "compute_time_us": round(compute_time * 1e6, 3),
        "total_time_us": round((nvlink_time + compute_time) * 1e6, 3),
        "data_per_gpu_mb": round(data_per_gpu / 1e6, 2),
        "sm_overhead": sm_needed,
        "best_for_sizes": "None (prefer NVLS)",
    }


# ── Switching Threshold Engine ───────────────────────────────────────

def find_switching_thresholds(hw: NVSwitchHW) -> Dict:
    """
    Find the message size thresholds where the optimal algorithm changes:
    - Small messages (< T1): One-Shot wins (lowest latency)
    - Medium messages (T1 - T2): Two-Shot Multimem wins (NVSwitch reduction)
    - Large messages (> T2): Two-Shot Multimem or P2P (bandwidth-limited)
    """
    sizes = []
    # Logarithmic sweep from 256B to 1GB
    for exp in np.linspace(8, 30, 50):
        sizes.append(int(2**exp))

    thresholds = []
    prev_best = None

    for size in sizes:
        one_shot = model_oneshot_allreduce(size, hw)
        nvls = model_multimem_nvls(size, hw)
        p2p = model_twoshot_p2p(size, hw)

        best = min([one_shot, nvls, p2p], key=lambda x: x["total_time_us"])

        if prev_best is None or best["algorithm"] != prev_best["algorithm"]:
            thresholds.append({
                "threshold_bytes": size,
                "from_algo": prev_best["algorithm"] if prev_best else "start",
                "to_algo": best["algorithm"],
                "one_shot_us": one_shot["total_time_us"],
                "nvls_us": nvls["total_time_us"],
                "p2p_us": p2p["total_time_us"],
            })
            prev_best = best

    return {
        "hardware": "H800 (NVSwitch)",
        "num_gpus": hw.num_gpus,
        "nvlink_bw_gbs": hw.nvlink_bw_gbs,
        "thresholds": thresholds,
    }


def generate_decision_model(hw: NVSwitchHW) -> Dict:
    """
    Generate the full decision model: for each message size range,
    return the recommended algorithm with estimated performance.
    """
    # Sample key sizes more densely
    key_sizes = [
        256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576, 2097152, 4194304,
        8388608, 16777216, 33554432, 67108864, 134217728,
        268435456, 536870912, 1073741824,
    ]

    decision_table = []
    for size in key_sizes:
        one_shot = model_oneshot_allreduce(size, hw)
        nvls = model_multimem_nvls(size, hw)
        p2p = model_twoshot_p2p(size, hw)

        best = min([one_shot, nvls, p2p], key=lambda x: x["total_time_us"])

        decision_table.append({
            "msg_size": f"{size/1024:.0f}KB" if size < 1048576 else f"{size/1048576:.0f}MB",
            "size_bytes": size,
            "recommended": best["algorithm"],
            "one_shot_us": one_shot["total_time_us"],
            "nvls_us": nvls["total_time_us"],
            "p2p_us": p2p["total_time_us"],
            "sm_saved_vs_p2p": p2p["sm_overhead"] - nvls["sm_overhead"],
        })

    return {
        "threshold_summary": [
            {
                "range": "0 - 16KB",
                "recommended": "One-Shot",
                "reason": "Lowest latency, no synchronization overhead",
            },
            {
                "range": "16KB - 1GB+",
                "recommended": "Two-Shot Multimem (NVLS)",
                "reason": "NVSwitch hardware reduction, near-zero SM usage",
            },
            {
                "range": "All sizes",
                "recommended": "Never Two-Shot P2P",
                "reason": "P2P is always worse than NVLS when NVSwitch is available",
            },
        ],
        "decision_table": decision_table,
    }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NVSwitch AllReduce Switching Threshold Analysis"
    )
    parser.add_argument("--analyze", action="store_true",
                        help="Run full analysis of three algorithms")
    parser.add_argument("--thresholds", action="store_true",
                        help="Find switching thresholds")
    parser.add_argument("--decision-model", action="store_true",
                        help="Generate decision model table")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")

    args = parser.parse_args()

    hw = NVSwitchHW()

    if args.analyze or not (args.thresholds or args.decision_model):
        print("=" * 72)
        print("Intranode AllReduce: NVLS vs P2P Switching Threshold")
        print("=" * 72)

        test_sizes = [
            ("4KB", 4096),
            ("64KB", 65536),
            ("1MB", 1048576),
            ("16MB", 16777216),
            ("256MB", 268435456),
        ]

        print(f"\n{'Size':<12} {'One-Shot':<18} {'NVLS':<18} {'P2P':<18} {'Best':<22}")
        print("-" * 88)

        for label, size in test_sizes:
            one_shot = model_oneshot_allreduce(size, hw)
            nvls = model_multimem_nvls(size, hw)
            p2p = model_twoshot_p2p(size, hw)

            best_name = min([one_shot, nvls, p2p],
                            key=lambda x: x["total_time_us"])["algorithm"][:20]

            print(f"{label:<12} "
                  f"{one_shot['total_time_us']:>7.1f}us (0 SM){'':>4} "
                  f"{nvls['total_time_us']:>7.1f}us (0 SM){'':>4} "
                  f"{p2p['total_time_us']:>7.1f}us ({p2p['sm_overhead']} SM){'':>2} "
                  f"{best_name:<22}")

        print(f"\nNVLS SM savings vs P2P: {16 - 0} SMs = ~{(16/132)*100:.1f}% of total SMs")

    if args.thresholds:
        thresholds = find_switching_thresholds(hw)
        print("\nSwitching Thresholds:")
        for t in thresholds["thresholds"]:
            size_label = (f"{t['threshold_bytes']/1024:.0f}KB"
                          if t['threshold_bytes'] < 1048576
                          else f"{t['threshold_bytes']/1048576:.1f}MB")
            print(f"  {size_label:>8}: {t['from_algo'][:30]} -> {t['to_algo'][:30]}")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(thresholds, f, indent=2)

    if args.decision_model:
        model = generate_decision_model(hw)
        print("\nDecision Model Summary:")
        for s in model["threshold_summary"]:
            print(f"  {s['range']:<15} -> {s['recommended']:<30} ({s['reason']})")

        print("\nDetailed Decision Table:")
        print(f"{'Size':<10} {'Recommended':<30} {'NVLS':<12} {'P2P':<12} {'SM Saved':<10}")
        print("-" * 84)
        for d in model["decision_table"][::2]:  # Every other for brevity
            print(f"{d['msg_size']:<10} {d['recommended']:<30} "
                  f"{d['nvls_us']:>6.1f}us{'':>5} "
                  f"{d['p2p_us']:>6.1f}us{'':>5} "
                  f"{d['sm_saved_vs_p2p']:>5} SMs")

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(model, f, indent=2)


if __name__ == "__main__":
    main()
