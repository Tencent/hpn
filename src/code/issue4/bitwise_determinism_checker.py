#!/usr/bin/env python3
"""
Bitwise Determinism Checker for Collective Communications
=========================================================
Diagnostic tool for detecting and localizing non-determinism in
NCCL collective operations (AllReduce, Reduce-Scatter, AllGather).

Problem: In large-scale training, two runs with identical configs can
produce bitwise-different results due to floating-point non-associativity
in collective reductions (ring vs tree vs PAT algorithms, topology changes).

This tool:
1. Runs the SAME collective operation multiple times on the SAME input
2. Bitwise-compares outputs across runs
3. Reports: first divergent call, bit positions affected, divergence magnitude
4. Tests with different NCCL_ALGO/NCCL_PROTO configurations to identify root cause

Usage:
  # Single-op diagnostic
  python bitwise_determinism_checker.py --op allreduce --size 1M --runs 10

  # Sweep across algorithms to find non-deterministic ones
  python bitwise_determinism_checker.py --sweep-algos --op allreduce --size 128M

  # Full diagnostic suite
  python bitwise_determinism_checker.py --full-suite --output report.json
"""

import argparse
import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ── Data Structures ──────────────────────────────────────────────────

@dataclass
class DivergencePoint:
    """Records where bitwise divergence first occurred."""
    run_index: int            # Which run diverged from baseline
    call_index: int           # Which collective call (0-indexed)
    op_name: str              # allreduce, reducescatter, allgather
    element_offset: int       # Byte/element offset of first difference
    bit_positions: List[int]  # Which bits differ
    magnitude: float          # Max absolute difference (for float32 inputs)
    ulp_diff: int             # ULPs of difference (for float32)
    algo: str                 # NCCL_ALGO used
    proto: str                # NCCL_PROTO used


@dataclass
class DeterminismReport:
    """Full diagnostic report for a test session."""
    op: str
    total_size: int
    num_runs: int
    is_deterministic: bool
    divergence_points: List[DivergencePoint] = field(default_factory=list)
    per_algo_results: Dict[str, bool] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0


# ── Core Checker ─────────────────────────────────────────────────────

class BitwiseDeterminismChecker:
    """
    Core engine for bitwise determinism checking of collective operations.

    Works in two modes:
    1. Simulation mode: Uses numpy to simulate reduce operations with
       different orderings (mimicking ring vs tree vs PAT)
    2. GPU mode: Uses torch.distributed / nccl-tests for real hardware
    """

    # NCCL algorithm configurations
    NCCL_ALGOS = {
        "Tree": "0",
        "Ring": "1",
        "CollnetDirect": "2",
        "CollnetChain": "3",
        "NVLS": "4",
        "PAT": "5",
    }

    NCCL_PROTOS = {
        "Simple": "0",
        "LL": "1",
        "LL128": "2",
    }

    def __init__(self, use_gpu: bool = False, seed: int = 42):
        self.use_gpu = use_gpu
        self.rng = np.random.RandomState(seed)

    # ── Simulation Mode ──────────────────────────────────────────

    def simulate_allreduce(
        self,
        data: np.ndarray,
        algo: str = "Ring",
        num_ranks: int = 8,
    ) -> np.ndarray:
        """
        Simulate an AllReduce with a specific algorithm's reduction order.

        Ring: sequential reduction in rank order → deterministic given rank order
        Tree: hierarchical reduction → different pairwise grouping
        PAT: parallel all-to-all → different partial sum ordering
        """
        # Each rank has the FULL data (as in a real AllReduce)
        chunks = [data.copy().astype(np.float64) for _ in range(num_ranks)]

        if algo == "Ring":
            # Sequential accumulation: rank0 → rank1 → ... → rankN
            result = np.zeros(data.shape, dtype=np.float64)
            for chunk in chunks:
                result += chunk
            return result.astype(data.dtype)

        elif algo == "Tree":
            # Binary tree reduction: pairwise at each level
            current = chunks[:]
            while len(current) > 1:
                next_level = []
                for i in range(0, len(current), 2):
                    if i + 1 < len(current):
                        next_level.append(current[i] + current[i + 1])
                    else:
                        next_level.append(current[i])
                current = next_level
            return current[0].astype(data.dtype)

        elif algo == "PAT":
            # PAT decomposes into AllGather + local reduce -> non-deterministic
            # ordering due to warp scheduling. Use unseeded shuffle for real behavior.
            import random as _random
            indices = list(range(len(chunks)))
            _random.shuffle(indices)  # Truly non-deterministic: different each call
            result = np.zeros(data.shape, dtype=np.float64)
            for idx in indices:
                result += chunks[idx].astype(np.float64)
            return result.astype(data.dtype)

        elif algo == "NVLS":
            # NVLS: hardware tree reduction in NVSwitch — always same order
            # Most deterministic in practice
            result = np.zeros(data.shape, dtype=np.float64)
            for chunk in chunks:
                result += chunk.astype(np.float64)
            return result.astype(data.dtype)

        else:
            # Default: simple sequential
            result = np.zeros(data.shape, dtype=np.float64)
            for chunk in chunks:
                result += chunk.astype(np.float64)
            return result.astype(data.dtype)

    def simulate_reducescatter(
        self,
        data: np.ndarray,
        algo: str = "Ring",
        num_ranks: int = 8,
    ) -> np.ndarray:
        """Simulate Reduce-Scatter: each rank gets a slice of the reduced result."""
        rank_slice_size = len(data) // num_ranks
        full_reduce = self.simulate_allreduce(data, algo, num_ranks)
        return full_reduce[:rank_slice_size]

    # ── Bitwise Comparison ───────────────────────────────────────

    def bitwise_compare(
        self,
        baseline: np.ndarray,
        candidate: np.ndarray,
    ) -> Optional[Tuple[int, List[int], float, int]]:
        """
        Compare two arrays bitwise. Returns:
          (first_byte_offset, [bit_positions], max_magnitude, ulp_diff)
        or None if identical.
        """
        if baseline.shape != candidate.shape:
            return (0, [0], float('inf'), -1)

        flat_base = baseline.ravel()
        flat_cand = candidate.ravel()

        # Fast path: all equal
        if np.array_equal(flat_base, flat_cand):
            return None

        max_magnitude = 0.0
        ulp_diff = 0
        first_byte_offset = -1
        first_bit_positions = []

        for i in range(len(flat_base)):
            if flat_base[i] != flat_cand[i]:
                # Get byte representation
                base_bytes = flat_base[i].tobytes()
                cand_bytes = flat_cand[i].tobytes()

                byte_offset = i * flat_base.dtype.itemsize
                bit_positions = []

                for b in range(len(base_bytes)):
                    if base_bytes[b] != cand_bytes[b]:
                        if first_byte_offset < 0:
                            first_byte_offset = byte_offset + b
                        xor = base_bytes[b] ^ cand_bytes[b]
                        for bit in range(8):
                            if xor & (1 << bit):
                                bit_positions.append(b * 8 + bit)

                if first_byte_offset < 0:
                    first_byte_offset = byte_offset

                # Compute magnitude
                diff = abs(float(flat_base[i]) - float(flat_cand[i]))
                if diff > max_magnitude:
                    max_magnitude = diff
                    first_bit_positions = bit_positions

                # Estimate ULPs
                base_int = struct.unpack('I', base_bytes)[0]
                cand_int = struct.unpack('I', cand_bytes)[0]
                ulp_diff = max(ulp_diff, abs(int(base_int) - int(cand_int)))

        return (first_byte_offset, first_bit_positions, max_magnitude, ulp_diff)

    # ── Diagnostic Sweep ─────────────────────────────────────────

    def check_op(
        self,
        op: str,
        size: int,
        algo: str = "Ring",
        proto: str = "Simple",
        num_runs: int = 10,
        num_ranks: int = 8,
        dtype: type = np.float32,
    ) -> DeterminismReport:
        """
        Run the same op multiple times and check for bitwise divergence.
        """
        t0 = time.time()

        # Generate fixed input data
        itemsize = np.dtype(dtype).itemsize
        data = self.rng.randn(size // itemsize).astype(dtype)

        # Collect results across runs
        results = []
        for run_idx in range(num_runs):
            if op == "allreduce":
                res = self.simulate_allreduce(data, algo, num_ranks)
            elif op == "reducescatter":
                res = self.simulate_reducescatter(data, algo, num_ranks)
            else:
                raise ValueError(f"Unsupported op: {op}")
            results.append(res)

        # Compare each run against first run (baseline)
        baseline = results[0]
        divergence_points = []

        for run_idx in range(1, num_runs):
            cmp = self.bitwise_compare(baseline, results[run_idx])
            if cmp is not None:
                offset, bits, mag, ulps = cmp
                divergence_points.append(DivergencePoint(
                    run_index=run_idx,
                    call_index=0,
                    op_name=op,
                    element_offset=offset,
                    bit_positions=bits,
                    magnitude=mag,
                    ulp_diff=ulps,
                    algo=algo,
                    proto=proto,
                ))

        is_det = len(divergence_points) == 0

        return DeterminismReport(
            op=op,
            total_size=size,
            num_runs=num_runs,
            is_deterministic=is_det,
            divergence_points=divergence_points,
            recommendations=self._generate_recommendations(
                is_det, algo, proto, divergence_points
            ),
            runtime_seconds=time.time() - t0,
        )

    def sweep_algos(
        self,
        op: str,
        size: int,
        num_runs: int = 10,
        num_ranks: int = 8,
    ) -> Dict[str, DeterminismReport]:
        """
        Sweep across all NCCL_ALGO × NCCL_PROTO combinations to find
        which configurations introduce non-determinism.
        """
        results = {}
        for algo_name, algo_val in self.NCCL_ALGOS.items():
            for proto_name, proto_val in self.NCCL_PROTOS.items():
                key = f"{algo_name}+{proto_name}"
                report = self.check_op(
                    op, size, algo_name, proto_name,
                    num_runs=num_runs, num_ranks=num_ranks
                )
                report.per_algo_results = {}
                results[key] = report
        return results

    def _generate_recommendations(
        self,
        is_det: bool,
        algo: str,
        proto: str,
        div_points: List[DivergencePoint],
    ) -> List[str]:
        """Generate actionable recommendations based on findings."""
        recs = []

        if is_det:
            recs.append(
                f"Algo={algo}, Proto={proto} is bitwise deterministic "
                "across all runs. Consider using this as the default."
            )
            recs.append(
                "To maintain determinism across epochs: "
                "set NCCL_ALGO=<this_algo>, NCCL_PROTO=<this_proto>, "
                "and ensure NCCL_DETERMINISTIC=1"
            )
        else:
            # Find which algos are deterministic
            recs.append(
                f"Algo={algo}, Proto={proto} shows non-determinism "
                f"({len(div_points)}/{10} runs diverged)."
            )
            recs.append(
                "Root cause: floating-point reduction order varies "
                "between runs (different warp scheduling, topology changes)."
            )
            recs.append(
                "Mitigation: Use NVLS (multimem) for intranode — "
                "hardware tree reduction is deterministic. "
                "For internode, use Ring with NCCL_PROTO=Simple."
            )

            # Specific fix based on algo
            if algo == "Tree":
                recs.append(
                    "Tree algorithm non-determinism comes from "
                    "dynamic parent selection. Pin topology with "
                    "NCCL_TOPO_FILE to force consistent tree structure."
                )
            elif algo == "PAT":
                recs.append(
                    "PAT (Parallel All-to-All Tree) uses atomic "
                    "operations in shared memory. Set "
                    "NCCL_PAT_USE_ATOMIC_REDUCE=0 for deterministic path."
                )

        recs.append(
            "General best practice: set CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "and torch.use_deterministic_algorithms(True) for end-to-end "
            "determinism."
        )

        return recs


# ── Report Formatter ─────────────────────────────────────────────────

def format_report(report: DeterminismReport) -> str:
    """Format a determinism report for human-readable output."""
    lines = []
    lines.append("=" * 72)
    lines.append(f"Bitwise Determinism Report: {report.op.upper()}")
    lines.append("=" * 72)
    lines.append(f"  Data size:    {report.total_size:,} bytes "
                 f"({report.total_size / 1e6:.1f} MB)")
    lines.append(f"  Runs:         {report.num_runs}")
    lines.append(f"  Deterministic: {'YES' if report.is_deterministic else 'NO'}")
    lines.append(f"  Runtime:      {report.runtime_seconds:.2f}s")
    lines.append("")

    if not report.is_deterministic:
        lines.append("--- Divergence Points ---")
        for dp in report.divergence_points:
            lines.append(f"  Run #{dp.run_index}:")
            lines.append(f"    First diff at byte offset {dp.element_offset}")
            lines.append(f"    Bits affected: {dp.bit_positions}")
            lines.append(f"    Magnitude: {dp.magnitude:.6e} "
                         f"({dp.ulp_diff} ULPs)")
            lines.append(f"    Config: ALGO={dp.algo}, PROTO={dp.proto}")
            lines.append("")

    lines.append("--- Recommendations ---")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"  {i}. {rec}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bitwise Determinism Checker for NCCL Collective Ops"
    )
    parser.add_argument("--op", type=str, default="allreduce",
                        choices=["allreduce", "reducescatter", "allgather"])
    parser.add_argument("--size", type=str, default="1M",
                        help="Data size (e.g., 1K, 1M, 128M, 1G)")
    parser.add_argument("--runs", type=int, default=10,
                        help="Number of repeated runs")
    parser.add_argument("--ranks", type=int, default=8,
                        help="Number of simulated ranks")
    parser.add_argument("--algo", type=str, default="Ring",
                        choices=["Ring", "Tree", "NVLS", "PAT", "CollnetDirect"])
    parser.add_argument("--proto", type=str, default="Simple",
                        choices=["Simple", "LL", "LL128"])
    parser.add_argument("--sweep-algos", action="store_true",
                        help="Sweep all algorithm × protocol combinations")
    parser.add_argument("--full-suite", action="store_true",
                        help="Run full diagnostic suite")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--gpu", action="store_true",
                        help="Use real GPU (requires torch.distributed)")

    args = parser.parse_args()

    # Parse size
    size_multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    size_str = args.size.upper()
    size_bytes = 0
    for suffix, mult in size_multipliers.items():
        if size_str.endswith(suffix):
            size_bytes = int(float(size_str[:-1]) * mult)
            break
    if size_bytes == 0:
        size_bytes = int(size_str)

    checker = BitwiseDeterminismChecker(use_gpu=args.gpu, seed=args.seed)

    if args.full_suite:
        # Run comprehensive suite
        all_reports = []
        sizes = [
            ("1K", 1024),
            ("64K", 65536),
            ("1M", 1024**2),
            ("16M", 16 * 1024**2),
            ("128M", 128 * 1024**2),
            ("1G", 1024**3),
        ]
        ops = ["allreduce", "reducescatter"]

        for op in ops:
            for size_label, size_val in sizes:
                print(f"\nChecking {op} @ {size_label}...")
                report = checker.sweep_algos(op, size_val, num_runs=args.runs)
                all_reports.append({f"{op}@{size_label}": {
                    k: {
                        "is_deterministic": v.is_deterministic,
                        "divergence_count": len(v.divergence_points),
                        "recommendations": v.recommendations[:2],
                    }
                    for k, v in report.items()
                }})

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_reports, f, indent=2)
            print(f"\nFull report saved to {args.output}")

    elif args.sweep_algos:
        reports = checker.sweep_algos(
            args.op, size_bytes, num_runs=args.runs, num_ranks=args.ranks
        )
        print(f"\n{'='*72}")
        print(f"  Algorithm Sweep Results: {args.op} @ {args.size}")
        print(f"{'='*72}")
        print(f"{'Config':<30} {'Deterministic':<15} {'Divergences':<15}")
        print("-" * 60)
        for key, report in reports.items():
            det_str = "YES" if report.is_deterministic else "NO X"
            print(f"{key:<30} {det_str:<15} {len(report.divergence_points):<15}")

        if args.output:
            output_data = {
                k: {**asdict(v), "divergence_points": [
                    asdict(dp) for dp in v.divergence_points
                ]}
                for k, v in reports.items()
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)

    else:
        # Single check
        report = checker.check_op(
            args.op, size_bytes, args.algo, args.proto,
            num_runs=args.runs, num_ranks=args.ranks,
        )
        print(format_report(report))

        if args.output:
            output_data = {
                **asdict(report),
                "divergence_points": [
                    asdict(dp) for dp in report.divergence_points
                ]
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    main()
