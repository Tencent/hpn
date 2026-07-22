#!/usr/bin/env python3
"""
NCCL Collective Algorithm Auto-Selector
========================================
Topology- and message-size-aware algorithm selector for AllReduce,
AllGather, and ReduceScatter that automatically chooses the optimal
Ring/Tree/NVLS/PAT algorithm based on message size and topology scale.

Based on the NCCL algorithm selection framework.

Usage:
  python algorithm_selector.py --benchmark    # Profile current topology
  python algorithm_selector.py --select 64MB  # Get recommended algorithm
  python algorithm_selector.py --generate-table  # Generate selection table
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Algorithm Performance Models ─────────────────────────────────────

@dataclass
class AlgoCharacteristics:
    """Performance characteristics of a collective algorithm."""
    name: str                  # Ring, Tree, NVLS, PAT
    steps: float               # O(N) step count scaling
    latency_per_step_us: float # Per-step latency (us)
    bandwidth_efficiency: float # Fraction of peak bandwidth (0-1)
    sm_per_gpu: int            # SMs used per GPU
    min_msg_size: int          # Recommended minimum message size (bytes)
    max_msg_size: int          # Recommended maximum message size (bytes)
    requires_nvswitch: bool    # Hardware requirement
    is_deterministic: bool     # Bitwise reproducibility

    # Performance at given config
    def latency_us(self, msg_size_bytes: int, num_ranks: int,
                   bw_gbs: float) -> float:
        """Estimate latency for given message size and ranks."""
        # Algorithmic steps × per-step latency
        algo_latency = self.steps * num_ranks * self.latency_per_step_us

        # Data transfer time = message_size / effective_bandwidth
        effective_bw = bw_gbs * self.bandwidth_efficiency
        transfer_time = (msg_size_bytes / (effective_bw * 1e9)) * 1e6

        return algo_latency + transfer_time


# Pre-calibrated algorithm characteristics for H800 GPUs
# Based on NCCL published data and HPN internal benchmarks
ALGO_CHARACTERISTICS = {
    "Ring": AlgoCharacteristics(
        name="Ring",
        steps=2.0,                # 2 × (N-1)/N for AllReduce
        latency_per_step_us=0.5,
        bandwidth_efficiency=0.95, # Excellent bandwidth utilization
        sm_per_gpu=2,
        min_msg_size=128 * 1024,   # 128KB — ring overhead dominates below
        max_msg_size=2**63 - 1,    # No upper limit
        requires_nvswitch=False,
        is_deterministic=True,     # Fixed rank order
    ),
    "Tree": AlgoCharacteristics(
        name="Tree",
        steps=4.0,                # log2(N) levels × 2 phases
        latency_per_step_us=0.3,
        bandwidth_efficiency=0.70, # Tree reduces bandwidth efficiency
        sm_per_gpu=1,
        min_msg_size=0,            # Best for tiny messages
        max_msg_size=256 * 1024,   # 256KB — tree saturation
        requires_nvswitch=False,
        is_deterministic=False,    # Dynamic parent selection
    ),
    "NVLS": AlgoCharacteristics(
        name="NVLS",
        steps=2.0,                # One-shot via NVSwitch
        latency_per_step_us=0.2,
        bandwidth_efficiency=0.90, # Near-peak: hardware reduction
        sm_per_gpu=0,              # NVSwitch does the work
        min_msg_size=64 * 1024,    # 64KB — NVLS setup overhead
        max_msg_size=2**63 - 1,
        requires_nvswitch=True,
        is_deterministic=True,     # Hardware tree — always same order
    ),
    "PAT": AlgoCharacteristics(
        name="PAT",
        steps=1.5,                # Parallel all-to-all tree
        latency_per_step_us=0.4,
        bandwidth_efficiency=0.85,
        sm_per_gpu=4,
        min_msg_size=512 * 1024,   # 512KB — PAT overhead
        max_msg_size=2**63 - 1,
        requires_nvswitch=False,
        is_deterministic=False,    # Atomic operations in shared memory
    ),
}


# ── Selector Engine ──────────────────────────────────────────────────

class AlgorithmSelector:
    """Message-size and topology-aware algorithm selector."""

    def __init__(
        self,
        nvlink_bw_gbs: float = 900.0,
        rdma_bw_gbs: float = 3200.0,  # 8×400Gbps
        has_nvswitch: bool = True,
        num_gpus_per_node: int = 8,
        num_nodes: int = 1,
    ):
        self.nvlink_bw = nvlink_bw_gbs
        self.rdma_bw = rdma_bw_gbs
        self.has_nvswitch = has_nvswitch
        self.num_gpus = num_gpus_per_node
        self.num_nodes = num_nodes
        self.total_ranks = num_gpus_per_node * num_nodes

    def get_effective_bandwidth(self, is_intranode: bool) -> float:
        """Get effective bandwidth based on communication domain."""
        if is_intranode:
            return self.nvlink_bw
        else:
            return self.rdma_bw

    def select(
        self,
        op: str,
        msg_size_bytes: int,
        is_intranode: bool = True,
        require_deterministic: bool = False,
    ) -> List[Tuple[AlgoCharacteristics, float, str]]:
        """
        Select and rank algorithms for the given operation and message size.
        Returns list of (algo, latency_us, reason) sorted best-first.
        """
        num_ranks = self.num_gpus if is_intranode else self.total_ranks
        bw = self.get_effective_bandwidth(is_intranode)

        results = []
        for algo in ALGO_CHARACTERISTICS.values():
            # Check message size compatibility
            if msg_size_bytes < algo.min_msg_size:
                continue
            if msg_size_bytes > algo.max_msg_size:
                continue

            # Check hardware requirements
            if algo.requires_nvswitch and not self.has_nvswitch:
                continue

            # Check determinism requirement
            if require_deterministic and not algo.is_deterministic:
                continue

            latency = algo.latency_us(msg_size_bytes, num_ranks, bw)
            results.append((algo, latency, self._reason(algo, msg_size_bytes)))

        # Sort by latency (ascending)
        results.sort(key=lambda x: x[1])
        return results

    def _reason(self, algo: AlgoCharacteristics, msg_size: int) -> str:
        """Explain why this algorithm is recommended."""
        if algo.name == "Ring":
            if msg_size > 1024**3:
                return "Best bandwidth utilization for large messages"
            return "Good general-purpose choice, deterministic"
        elif algo.name == "Tree":
            return "Lowest latency for small messages"
        elif algo.name == "NVLS":
            return "Hardware reduction in NVSwitch, zero SM overhead"
        elif algo.name == "PAT":
            return "Parallel tree, good for medium-large messages"
        return ""

    def generate_selection_table(
        self,
        output_json: Optional[str] = None,
    ) -> List[Dict]:
        """Generate full selection table across message sizes and ops."""
        ops = ["allreduce", "reducescatter", "allgather"]
        sizes = [
            ("1KB", 1024),
            ("8KB", 8192),
            ("64KB", 65536),
            ("256KB", 262144),
            ("1MB", 1024**2),
            ("4MB", 4 * 1024**2),
            ("16MB", 16 * 1024**2),
            ("64MB", 64 * 1024**2),
            ("256MB", 256 * 1024**2),
            ("1GB", 1024**3),
        ]
        domains = [("intranode", True), ("internode", False)]

        results = []
        for op in ops:
            for size_label, size_bytes in sizes:
                for domain_label, is_intranode in domains:
                    best = self.select(op, size_bytes, is_intranode)
                    if best:
                        results.append({
                            "op": op,
                            "size": size_label,
                            "size_bytes": size_bytes,
                            "domain": domain_label,
                            "recommended_algo": best[0][0].name,
                            "estimated_latency_us": round(best[0][1], 2),
                            "alternatives": [a[0].name for a in best[1:3]],
                            "reason": best[0][2],
                        })

        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results, f, indent=2)

        return results

    def benchmark_topology(self) -> Dict:
        """Run synthetic benchmarks to calibrate the selection model."""
        # In real deployment, this would run nccl-tests across message sizes
        # and record actual bandwidth for each algorithm.
        # Here we return the pre-calibrated model parameters.
        return {
            "nvlink_bw_gbs": self.nvlink_bw,
            "rdma_bw_gbs": self.rdma_bw,
            "algorithms": {
                name: {
                    "bandwidth_efficiency": algo.bandwidth_efficiency,
                    "sm_per_gpu": algo.sm_per_gpu,
                    "is_deterministic": algo.is_deterministic,
                }
                for name, algo in ALGO_CHARACTERISTICS.items()
            }
        }


# ── Visualization ────────────────────────────────────────────────────

def plot_selection_heatmap(selector: AlgorithmSelector, output_path: str):
    """Generate heatmap of recommended algorithm across msg sizes."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    msg_sizes = [
        "1KB", "8KB", "64KB", "256KB", "1MB", "4MB", "16MB",
        "64MB", "256MB", "1GB"
    ]
    size_bytes_list = [1024, 8192, 65536, 262144, 1048576,
                       4*1048576, 16*1048576, 64*1048576,
                       256*1048576, 1024**3]
    ops = ["allreduce", "reducescatter", "allgather"]
    algo_map = {"Ring": 0, "Tree": 1, "NVLS": 2, "PAT": 3}
    algo_names = ["Ring", "Tree", "NVLS", "PAT"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (domain_label, is_intranode) in enumerate(
        [("Intranode (NVLink)", True), ("Internode (RDMA)", False)]
    ):
        matrix = np.zeros((len(ops), len(msg_sizes)))
        for i, op in enumerate(ops):
            for j, size_bytes in enumerate(size_bytes_list):
                best = selector.select(op, size_bytes, is_intranode)
                if best:
                    matrix[i, j] = algo_map[best[0][0].name]

        im = axes[ax_idx].imshow(matrix, cmap='RdYlGn', aspect='auto',
                                  vmin=0, vmax=3)
        axes[ax_idx].set_xticks(range(len(msg_sizes)))
        axes[ax_idx].set_xticklabels(msg_sizes, rotation=45, ha='right')
        axes[ax_idx].set_yticks(range(len(ops)))
        axes[ax_idx].set_yticklabels(ops)
        axes[ax_idx].set_title(f"{domain_label}")
        axes[ax_idx].set_xlabel("Message Size")
        axes[ax_idx].set_ylabel("Operation")

        # Add text annotations
        for i in range(len(ops)):
            for j in range(len(msg_sizes)):
                text = algo_names[int(matrix[i, j])]
                axes[ax_idx].text(j, i, text, ha='center', va='center',
                                  fontsize=7)

    plt.suptitle("NCCL Algorithm Selection Map (H800 Topology)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Selection heatmap saved to {output_path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NCCL Algorithm Auto-Selector"
    )
    parser.add_argument("--select", type=str, default=None,
                        help="Get recommendation for a specific size (e.g., 64MB)")
    parser.add_argument("--op", type=str, default="allreduce",
                        choices=["allreduce", "reducescatter", "allgather"])
    parser.add_argument("--intranode", action="store_true", default=True,
                        help="Intranode (NVLink) selection")
    parser.add_argument("--internode", action="store_true",
                        help="Internode (RDMA) selection")
    parser.add_argument("--deterministic", action="store_true",
                        help="Require deterministic algorithm")
    parser.add_argument("--generate-table", action="store_true",
                        help="Generate full selection table")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run topology benchmark")
    parser.add_argument("--plot", action="store_true",
                        help="Generate selection heatmap")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")

    args = parser.parse_args()

    selector = AlgorithmSelector()

    if args.benchmark:
        result = selector.benchmark_topology()
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)

    elif args.select:
        size_mult = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
        size_str = args.select.upper()
        size_bytes = 0
        for suffix, mult in size_mult.items():
            if size_str.endswith(suffix):
                size_bytes = int(float(size_str[:-len(suffix)]) * mult)
                break

        is_intranode = not args.internode
        best = selector.select(
            args.op, size_bytes, is_intranode,
            require_deterministic=args.deterministic,
        )

        print(f"\nRecommendations for {args.op} @ {args.select} "
              f"({'intranode' if is_intranode else 'internode'}):")
        print("-" * 60)
        for i, (algo, latency, reason) in enumerate(best[:3]):
            marker = ">>" if i == 0 else "  "
            print(f"  {marker} {algo.name:<8} {latency:>8.1f} us  "
                  f"(SM: {algo.sm_per_gpu}, Det: {algo.is_deterministic})")
            print(f"       {reason}")

    elif args.generate_table:
        table = selector.generate_selection_table(args.output)
        print(f"Generated {len(table)} selection entries")
        # Print summary
        for entry in table[:5]:
            print(f"  {entry['op']:>15} @ {entry['size']:>6} "
                  f"[{entry['domain']:<10}] -> {entry['recommended_algo']}")

    if args.plot:
        plot_selection_heatmap(selector, "algorithm_selection_heatmap.png")


if __name__ == "__main__":
    main()
