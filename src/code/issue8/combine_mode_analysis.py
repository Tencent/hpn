#!/usr/bin/env python3
"""
MoE Combine Three-Mode Precision/Traffic Trade-off Analysis
============================================================
Analytical model for DeepEP V2's three combine reduction modes
based on the kernel implementations in:
  deep_ep/include/deep_ep/impls/combine.cuh
  deep_ep/include/deep_ep/impls/combine_reduce_epilogue.cuh
  deep_ep/include/deep_ep/impls/combine_utils.cuh

Three modes (controlled by template params in combine_impl):
  Mode A: kUseExpandedLayout=false, kAllowMultipleReduction=false → no local reduce
  Mode B: kUseExpandedLayout=true,  kAllowMultipleReduction=true  → local reduction
  Mode C: kUseExpandedLayout=true,  kAllowMultipleReduction=false → expanded send
"""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Hardware Parameters ──────────────────────────────────────────────

@dataclass
class HardwareParams:
    """Configurable hardware parameters for the analytical model."""
    # Network bandwidths (GB/s)
    nvlink_bw: float = 900.0       # NVLink per-GPU bandwidth (H800: 900 GB/s)
    rdma_bw: float = 400.0         # RDMA per-rail bandwidth (400Gbps RDMA)
    num_rdma_rails: int = 8        # Number of RDMA rails (H800: 8× CX7)

    # Computation
    sm_clock_ghz: float = 1.98     # SM clock (H800: 1980 MHz)
    num_sms: int = 132             # Total SMs (H800: 132)

    # Memory
    hbm_bw: float = 3350.0         # HBM bandwidth (H800: 3.35 TB/s)
    shared_mem_per_sm: int = 227   # Shared memory per SM in KB

    # BF16 throughput (TFLOPS)
    bf16_tflops: float = 990.0     # H800 BF16: 990 TFLOPS

    @property
    def nvlink_total_bw(self) -> float:
        return self.nvlink_bw

    @property
    def rdma_total_bw(self) -> float:
        return self.rdma_bw * self.num_rdma_rails


# ── Workload Parameters ──────────────────────────────────────────────

@dataclass
class WorkloadParams:
    """Workload configuration for MoE combine."""
    num_experts: int = 64          # Total number of experts
    num_topk: int = 8              # Top-k routing
    hidden_dim: int = 7168         # Hidden dimension
    num_tokens: int = 4096         # Number of tokens per rank
    num_scaleout_ranks: int = 4    # Scale-out (inter-node) ranks
    num_scaleup_ranks: int = 8     # Scale-up (intra-node) ranks
    topk_repetition_rate: float = 0.3  # Fraction of tokens with duplicate top-k assignments

    @property
    def num_ranks(self) -> int:
        return (self.num_scaleout_ranks if self.num_scaleout_ranks > 1
                else self.num_scaleup_ranks)

    @property
    def bytes_per_token(self) -> int:
        return self.hidden_dim * 2  # BF16 = 2 bytes

    @property
    def total_bytes(self) -> int:
        return self.num_tokens * self.bytes_per_token


# ── Mode Analysis ────────────────────────────────────────────────────

@dataclass
class ModeResult:
    """Analysis result for a single combine mode."""
    mode_name: str
    mode_label: str                # A, B, or C

    # Traffic analysis
    network_traffic_gb: float      # Total network traffic in GB
    hbm_read_gb: float             # HBM read traffic
    hbm_write_gb: float            # HBM write traffic

    # Latency estimates
    network_latency_us: float      # Network transfer time
    compute_latency_us: float      # Reduction compute time
    total_latency_us: float        # Total combine time

    # Precision
    expected_precision_bits: float # Effective mantissa bits preserved
    max_accumulation_error: float  # Worst-case accumulation error (relative)

    # SM utilization
    sm_utilization_pct: float      # Percentage of SMs used
    sm_count_used: int             # Absolute SM count


def analyze_combine_mode(
    hw: HardwareParams,
    wl: WorkloadParams,
    mode: str
) -> ModeResult:
    """Analyze a single combine mode given hardware and workload."""

    bytes_per_elem = 2  # BF16
    hidden = wl.hidden_dim

    if mode == "A":
        # Mode A: No-expand, no local reduce
        # One token maps to exactly one source rank
        # Traffic = num_tokens × hidden bytes (one copy each)
        network_traffic_bytes = wl.num_tokens * hidden * bytes_per_elem
        hbm_read = network_traffic_bytes
        hbm_write = network_traffic_bytes

        # No reduction computation — just TMA load + store
        # TMA bandwidth limited, not compute limited
        sm_used = 4  # Minimal SMs for TMA orchestration
        compute_us = 0.0
        network_us = (network_traffic_bytes / (hw.rdma_total_bw * 1e9)) * 1e6
        # Scale-out: uses RDMA; Scale-up: uses NVLink
        if wl.num_scaleout_ranks > 1:
            network_us = max(
                network_traffic_bytes / (hw.rdma_total_bw * 1e9) * 1e6,
                network_traffic_bytes / (hw.nvlink_total_bw * 1e9) * 1e6
            )

        # Precision: best case, no intermediate reduction
        precision_bits = 7.0  # BF16 mantissa
        max_error = 0.0  # No accumulation error

    elif mode == "B":
        # Mode B: Expand + allow multiple reduction
        # Tokens with duplicate assignments are locally reduced before send
        # Traffic reduction proportional to (1 - repetition_rate)
        dedup_factor = 1.0 - wl.topk_repetition_rate * (1.0 - 1.0 / wl.num_topk)
        effective_tokens = int(wl.num_tokens * dedup_factor)
        network_traffic_bytes = effective_tokens * hidden * bytes_per_elem

        # HBM: read all (topk copies), write reduced
        hbm_read = wl.num_tokens * wl.num_topk * hidden * bytes_per_elem * wl.topk_repetition_rate
        hbm_write = network_traffic_bytes

        # Compute: shared memory vectorized BF16 reduction
        # Each duplicate requires (hidden / 32) BF16 adds per lane
        avg_duplicates = 1.0 + wl.topk_repetition_rate * (wl.num_topk - 1)
        reduction_ops = (wl.num_tokens * wl.topk_repetition_rate *
                         hidden / 2)  # BF162 adds
        compute_us = reduction_ops / (hw.bf16_tflops * 1e12) * 1e6 * 2  # factor 2 for read+add

        # HBM load latency dominates
        hbm_load_us = hbm_read / (hw.hbm_bw * 1e9) * 1e6
        network_us = network_traffic_bytes / (hw.rdma_total_bw * 1e9) * 1e6
        if wl.num_scaleout_ranks > 1:
            network_us = max(network_us,
                           network_traffic_bytes / (hw.nvlink_total_bw * 1e9) * 1e6)

        total_us = max(hbm_load_us, network_us) + compute_us
        sm_used = min(hw.num_sms, 24)  # Moderate SM usage

        # Precision: BF16 accumulation — each reduction loses ~1 ULP
        precision_bits = 7.0 - 0.5 * math.log2(max(1, avg_duplicates))
        max_error = (avg_duplicates - 1) * 2**-7  # BF16 epsilon

    elif mode == "C":
        # Mode C: Expanded send, no local reduce
        # Each top-k copy sent independently, epilogue reduces in FP32
        network_traffic_bytes = wl.num_tokens * wl.num_topk * hidden * bytes_per_elem

        hbm_read = network_traffic_bytes
        hbm_write = wl.num_tokens * hidden * bytes_per_elem  # Final write

        # Network limited — multiple copies over RDMA
        network_us = network_traffic_bytes / (hw.rdma_total_bw * 1e9) * 1e6
        if wl.num_scaleout_ranks > 1:
            network_us = max(network_us,
                           network_traffic_bytes / (hw.nvlink_total_bw * 1e9) * 1e6)

        # FP32 epilogue reduction
        epilogue_ops = wl.num_tokens * wl.num_topk * hidden
        compute_us = epilogue_ops / (hw.bf16_tflops * 1e12) * 1e6

        sm_used = min(hw.num_sms, 64)  # Higher SM for epilogue

        # Precision: FP32 accumulation — excellent
        precision_bits = 23.0  # FP32 mantissa
        max_error = (wl.num_topk - 1) * 2**-23  # FP32 epsilon × topk

    else:
        raise ValueError(f"Unknown mode: {mode}")

    total_us = network_us + compute_us

    return ModeResult(
        mode_name=f"Mode {mode}",
        mode_label=mode,
        network_traffic_gb=network_traffic_bytes / 1e9,
        hbm_read_gb=hbm_read / 1e9,
        hbm_write_gb=hbm_write / 1e9,
        network_latency_us=round(network_us, 3),
        compute_latency_us=round(compute_us, 3),
        total_latency_us=round(total_us, 3),
        expected_precision_bits=round(precision_bits, 2),
        max_accumulation_error=round(max_error, 8),
        sm_utilization_pct=round(sm_used / hw.num_sms * 100, 1),
        sm_count_used=sm_used,
    )


def compare_modes(
    hw: HardwareParams,
    wl: WorkloadParams
) -> List[ModeResult]:
    """Compare all three combine modes for given parameters."""
    return [
        analyze_combine_mode(hw, wl, "A"),
        analyze_combine_mode(hw, wl, "B"),
        analyze_combine_mode(hw, wl, "C"),
    ]


# ── Decision Model ───────────────────────────────────────────────────

def generate_decision_table(
    hw: HardwareParams,
    output_json: Optional[str] = None
) -> List[Dict]:
    """
    Generate a comprehensive decision table across varying:
    - Top-k repetition rates: 0.0 to 1.0
    - Message sizes (num_tokens): 256 to 16384
    - Precision requirements: "relaxed", "standard", "strict"
    """
    results = []
    repetition_rates = np.linspace(0.0, 1.0, 11)
    token_counts = [256, 512, 1024, 2048, 4096, 8192, 16384]
    hidden_dims = [4096, 7168, 14336]

    for rr in repetition_rates:
        for nt in token_counts:
            for hd in hidden_dims:
                wl = WorkloadParams(
                    num_tokens=nt,
                    hidden_dim=hd,
                    topk_repetition_rate=float(rr),
                )
                modes = compare_modes(hw, wl)

                # Determine best mode by:
                # 1. If strict precision needed → prefer C (FP32) or A (no reduction)
                # 2. If bandwidth-limited → prefer B (reduced traffic)
                # 3. Otherwise → A (simplest, lowest latency)

                best_for_traffic = min(modes, key=lambda m: m.network_traffic_gb)
                best_for_latency = min(modes, key=lambda m: m.total_latency_us)
                best_for_precision = max(modes, key=lambda m: m.expected_precision_bits)

                results.append({
                    "topk_repetition_rate": round(float(rr), 2),
                    "num_tokens": nt,
                    "hidden_dim": hd,
                    "total_bytes": wl.total_bytes,
                    "mode_A_traffic_gb": round(modes[0].network_traffic_gb, 4),
                    "mode_A_latency_us": modes[0].total_latency_us,
                    "mode_A_precision_bits": modes[0].expected_precision_bits,
                    "mode_B_traffic_gb": round(modes[1].network_traffic_gb, 4),
                    "mode_B_latency_us": modes[1].total_latency_us,
                    "mode_B_precision_bits": modes[1].expected_precision_bits,
                    "mode_C_traffic_gb": round(modes[2].network_traffic_gb, 4),
                    "mode_C_latency_us": modes[2].total_latency_us,
                    "mode_C_precision_bits": modes[2].expected_precision_bits,
                    "best_for_traffic": best_for_traffic.mode_label,
                    "best_for_latency": best_for_latency.mode_label,
                    "best_for_precision": best_for_precision.mode_label,
                    "recommended_mode": _recommend_mode(modes, rr),
                })

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

    return results


def _recommend_mode(modes: List[ModeResult], rr: float) -> str:
    """Recommend a combine mode based on trade-off analysis."""
    a, b, c = modes

    # Low repetition: mode A is always best (no overhead)
    if rr < 0.1:
        return "A"

    # High repetition + large traffic → mode B saves bandwidth
    if rr > 0.3 and b.network_traffic_gb < c.network_traffic_gb * 0.7:
        if b.expected_precision_bits >= 5.0:  # At least 5 bits precision
            return "B"

    # Strict precision needed → mode C (FP32)
    if rr > 0.5 and c.expected_precision_bits > b.expected_precision_bits + 10:
        return "C"

    # Default: pick best latency/precision trade-off
    # Mode B is generally the sweet spot for MoE
    if b.total_latency_us < a.total_latency_us * 1.1:
        return "B"

    return "A"


# ── Visualization ────────────────────────────────────────────────────

def plot_comparison(hw: HardwareParams, output_path: str = "combine_mode_comparison.png"):
    """Generate comparison plots for the three combine modes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot generation")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    repetition_rates = np.linspace(0.0, 1.0, 50)
    wl = WorkloadParams(num_tokens=4096, hidden_dim=7168)

    traffic_data = {"A": [], "B": [], "C": []}
    latency_data = {"A": [], "B": [], "C": []}
    precision_data = {"A": [], "B": [], "C": []}

    for rr in repetition_rates:
        wl.topk_repetition_rate = float(rr)
        modes = compare_modes(hw, wl)
        for mode in modes:
            traffic_data[mode.mode_label].append(mode.network_traffic_gb)
            latency_data[mode.mode_label].append(mode.total_latency_us)
            precision_data[mode.mode_label].append(mode.expected_precision_bits)

    # Traffic vs repetition rate
    ax = axes[0, 0]
    for label, color in [("A", "#2196F3"), ("B", "#4CAF50"), ("C", "#FF5722")]:
        ax.plot(repetition_rates, traffic_data[label], color=color, label=f"Mode {label}", linewidth=2)
    ax.set_xlabel("Top-K Repetition Rate")
    ax.set_ylabel("Network Traffic (GB)")
    ax.set_title("Network Traffic vs Repetition Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Latency vs repetition rate
    ax = axes[0, 1]
    for label, color in [("A", "#2196F3"), ("B", "#4CAF50"), ("C", "#FF5722")]:
        ax.plot(repetition_rates, latency_data[label], color=color, label=f"Mode {label}", linewidth=2)
    ax.set_xlabel("Top-K Repetition Rate")
    ax.set_ylabel("Total Latency (μs)")
    ax.set_title("Combine Latency vs Repetition Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision vs repetition rate
    ax = axes[1, 0]
    for label, color in [("A", "#2196F3"), ("B", "#4CAF50"), ("C", "#FF5722")]:
        ax.plot(repetition_rates, precision_data[label], color=color, label=f"Mode {label}", linewidth=2)
    ax.set_xlabel("Top-K Repetition Rate")
    ax.set_ylabel("Effective Precision (mantissa bits)")
    ax.set_title("Numerical Precision vs Repetition Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Decision regions
    ax = axes[1, 1]
    token_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    rr_vals = np.linspace(0.0, 1.0, 30)

    # Build decision matrix
    decision_map = np.zeros((len(token_sizes), len(rr_vals)), dtype=int)
    for i, nt in enumerate(token_sizes):
        for j, rr in enumerate(rr_vals):
            wl_i = WorkloadParams(num_tokens=nt, hidden_dim=7168, topk_repetition_rate=float(rr))
            modes = compare_modes(hw, wl_i)
            rec = _recommend_mode(modes, float(rr))
            decision_map[i, j] = ord(rec) - ord('A')

    cmap = plt.cm.RdYlGn
    im = ax.pcolormesh(rr_vals, token_sizes, decision_map,
                       cmap=cmap, vmin=0, vmax=2, shading='auto')
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.set_ticklabels(['Mode A\n(No Expand)', 'Mode B\n(Reduce)', 'Mode C\n(Expanded)'])
    ax.set_xlabel("Top-K Repetition Rate")
    ax.set_ylabel("Number of Tokens")
    ax.set_title("Recommended Combine Mode")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MoE Combine Three-Mode Precision/Traffic Trade-off Analysis"
    )
    parser.add_argument("--experts", type=int, default=64, help="Number of experts")
    parser.add_argument("--topk", type=int, default=8, help="Top-k")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension")
    parser.add_argument("--tokens", type=int, default=4096, help="Number of tokens")
    parser.add_argument("--repetition-rate", type=float, default=0.3,
                        help="Top-k repetition rate (0.0-1.0)")
    parser.add_argument("--scaleout-ranks", type=int, default=4, help="Scale-out ranks")
    parser.add_argument("--scaleup-ranks", type=int, default=8, help="Scale-up ranks")
    parser.add_argument("--generate-decision-table", action="store_true",
                        help="Generate full decision table as JSON")
    parser.add_argument("--plot", action="store_true", help="Generate comparison plots")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for decision table JSON")

    args = parser.parse_args()

    hw = HardwareParams()
    wl = WorkloadParams(
        num_experts=args.experts,
        num_topk=args.topk,
        hidden_dim=args.hidden,
        num_tokens=args.tokens,
        num_scaleout_ranks=args.scaleout_ranks,
        num_scaleup_ranks=args.scaleup_ranks,
        topk_repetition_rate=args.repetition_rate,
    )

    print("=" * 72)
    print("MoE Combine Mode Analysis — DeepEP V2")
    print("=" * 72)
    print(f"\nWorkload: experts={wl.num_experts}, topk={wl.num_topk}, "
          f"hidden={wl.hidden_dim}, tokens={wl.num_tokens}")
    print(f"Topology: scaleout={wl.num_scaleout_ranks}, scaleup={wl.num_scaleup_ranks}")
    print(f"Repetition rate: {wl.topk_repetition_rate:.2f}")
    print()

    modes = compare_modes(hw, wl)

    # Header
    print(f"{'':<12} {'Mode A':<20} {'Mode B':<20} {'Mode C':<20}")
    print(f"{'':<12} {'(No Expand)':<20} {'(Local Reduce)':<20} {'(Expanded Send)':<20}")
    print("-" * 72)

    # Traffic
    print(f"{'Net Traffic':<12} {modes[0].network_traffic_gb:>8.4f} GB{'':>8} "
          f"{modes[1].network_traffic_gb:>8.4f} GB{'':>8} "
          f"{modes[2].network_traffic_gb:>8.4f} GB{'':>8}")

    # Latency
    print(f"{'Net Latency':<12} {modes[0].network_latency_us:>9.1f} μs{'':>7} "
          f"{modes[1].network_latency_us:>9.1f} μs{'':>7} "
          f"{modes[2].network_latency_us:>9.1f} μs{'':>7}")

    print(f"{'Comp Latency':<12} {modes[0].compute_latency_us:>9.1f} μs{'':>7} "
          f"{modes[1].compute_latency_us:>9.1f} μs{'':>7} "
          f"{modes[2].compute_latency_us:>9.1f} μs{'':>7}")

    print(f"{'Total Latency':<12} {modes[0].total_latency_us:>9.1f} μs{'':>7} "
          f"{modes[1].total_latency_us:>9.1f} μs{'':>7} "
          f"{modes[2].total_latency_us:>9.1f} μs{'':>7}")

    # Precision
    print(f"{'Precision':<12} {modes[0].expected_precision_bits:>8.1f} bits{'':>7} "
          f"{modes[1].expected_precision_bits:>8.1f} bits{'':>7} "
          f"{modes[2].expected_precision_bits:>8.1f} bits{'':>7}")

    print(f"{'Max Error':<12} {modes[0].max_accumulation_error:>8.2e}{'':>8} "
          f"{modes[1].max_accumulation_error:>8.2e}{'':>8} "
          f"{modes[2].max_accumulation_error:>8.2e}{'':>8}")

    # SM
    print(f"{'SM Usage':<12} {modes[0].sm_utilization_pct:>8.1f}% ({modes[0].sm_count_used} SMs){'':>1} "
          f"{modes[1].sm_utilization_pct:>8.1f}% ({modes[1].sm_count_used} SMs){'':>1} "
          f"{modes[2].sm_utilization_pct:>8.1f}% ({modes[2].sm_count_used} SMs){'':>1}")

    print()
    rec = _recommend_mode(modes, wl.topk_repetition_rate)
    print(f"Recommended mode for repetition_rate={wl.topk_repetition_rate:.2f}: "
          f"\033[1;32mMode {rec}\033[0m")

    if args.generate_decision_table:
        print("\n" + "=" * 72)
        print("Generating full decision table...")
        output_file = args.output or "combine_decision_table.json"
        table = generate_decision_table(hw, output_file)
        print(f"Decision table with {len(table)} entries saved to {output_file}")

    if args.plot:
        plot_comparison(hw)


if __name__ == "__main__":
    main()
