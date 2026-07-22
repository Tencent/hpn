#!/usr/bin/env python3
"""
Multi-Table AlltoAllv Aggregation Fusion Operator
==================================================
Analytical model and framework for fusing multiple embedding table
AlltoAllv communications into a single aggregated operation.

In recommendation/advertising training, sparse embeddings span many tables,
each triggering a separate AlltoAllv. This causes:
  - High kernel launch count
  - Control-plane negotiation overhead per table
  - Small messages that under-utilize bandwidth

Solution: Aggregate multi-table AlltoAllv into one fused communication.

Usage:
  python multi_table_alltoall_fusion.py --num-tables 16 --tokens-per-table 1024
  python multi_table_alltoall_fusion.py --benchmark-model
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Models ────────────────────────────────────────────────────────────

@dataclass
class TableConfig:
    """Configuration for a single embedding table."""
    name: str
    embedding_dim: int       # Hidden dim per embedding
    num_tokens: int          # Tokens processed by this table
    first_dim: int           # Feature dimension (vocab partition size)

    @property
    def bytes_per_exchange(self) -> int:
        return self.num_tokens * self.embedding_dim * 2  # BF16


@dataclass
class AlltoAllResult:
    """Result of a single or fused AlltoAllv operation."""
    method: str                     # "separate" or "fused"
    num_tables: int
    total_bytes: int
    num_launches: int
    control_overhead_us: float      # Control-plane negotiation time
    data_transfer_us: float         # Data-plane transfer time
    total_time_us: float
    effective_bw_gbs: float


# ── Analytical Model ─────────────────────────────────────────────────

def model_separate_alltoall(
    tables: List[TableConfig],
    bw_gbs: float = 400.0,
    launch_overhead_us: float = 5.0,
    negotiation_overhead_us: float = 3.0,
) -> AlltoAllResult:
    """
    Model: each table performs its own AlltoAllv independently.
    Total = sum of individual operations + N × launch overhead.
    """
    total_bytes = 0
    total_data_time = 0.0

    for tbl in tables:
        b = tbl.bytes_per_exchange
        total_bytes += b
        total_data_time += b / (bw_gbs * 1e9)

    num_launches = len(tables)
    control_time = num_launches * (launch_overhead_us + negotiation_overhead_us)
    data_time_us = total_data_time * 1e6
    total_us = control_time + data_time_us

    return AlltoAllResult(
        method="separate",
        num_tables=num_launches,
        total_bytes=total_bytes,
        num_launches=num_launches,
        control_overhead_us=round(control_time, 3),
        data_transfer_us=round(data_time_us, 3),
        total_time_us=round(total_us, 3),
        effective_bw_gbs=round(total_bytes / (total_us * 1e-6) / 1e9, 2),
    )


def model_fused_alltoall(
    tables: List[TableConfig],
    bw_gbs: float = 400.0,
    launch_overhead_us: float = 5.0,
    negotiation_overhead_us: float = 3.0,
    merge_overhead_us: float = 2.0,
) -> AlltoAllResult:
    """
    Model: all tables aggregated into one fused AlltoAllv.
    - Merge all send/recv buffers into a single super-buffer
    - Single kernel launch
    - Single control-plane negotiation
    - Additional merge overhead proportional to number of tables
    """
    total_bytes = sum(t.bytes_per_exchange for t in tables)

    # Merge overhead: compute unified split/displacement arrays
    merge_time = merge_overhead_us * len(tables)

    # Single data transfer at full bandwidth
    data_time_us = total_bytes / (bw_gbs * 1e9) * 1e6

    # Single launch + negotiation
    control_time = launch_overhead_us + negotiation_overhead_us
    total_us = control_time + merge_time + data_time_us

    return AlltoAllResult(
        method="fused",
        num_tables=len(tables),
        total_bytes=total_bytes,
        num_launches=1,
        control_overhead_us=round(control_time, 3),
        data_transfer_us=round(data_time_us, 3),
        total_time_us=round(total_us, 3),
        effective_bw_gbs=round(total_bytes / (total_us * 1e-6) / 1e9, 2),
    )


# ── Fusion Strategy ─────────────────────────────────────────────────

def compute_optimal_fusion_strategy(
    tables: List[TableConfig],
    bw_gbs: float = 400.0,
    max_tables_per_group: int = 64,
) -> Dict:
    """
    Determine the optimal fusion strategy:
    1. Fuse ALL tables into one call (maximum aggregation)
    2. Group tables by embedding_dim (size-aligned groups)
    3. Fuse only small tables (below threshold), large ones separately

    Returns analysis of each strategy.
    """
    strategies = {}

    # Strategy 1: Full fusion
    strategies["full_fusion"] = model_fused_alltoall(tables, bw_gbs)

    # Strategy 2: Group by embedding dim
    dim_groups = {}
    for t in tables:
        dim_groups.setdefault(t.embedding_dim, []).append(t)

    group_results = []
    for dim, group in dim_groups.items():
        if len(group) == 1:
            group_results.append(model_separate_alltoall(group, bw_gbs))
        else:
            group_results.append(model_fused_alltoall(group, bw_gbs))

    total_dim_group_us = sum(r.total_time_us for r in group_results)
    strategies["dim_grouped"] = {
        "num_groups": len(dim_groups),
        "total_launches": sum(r.num_launches for r in group_results),
        "total_time_us": round(total_dim_group_us, 3),
    }

    # Strategy 3: Threshold-based (small tables fused, large separate)
    threshold_bytes = 64 * 1024  # 64KB threshold
    small_tables = [t for t in tables if t.bytes_per_exchange < threshold_bytes]
    large_tables = [t for t in tables if t.bytes_per_exchange >= threshold_bytes]

    small_result = model_fused_alltoall(small_tables, bw_gbs) if small_tables else None
    large_results = [model_separate_alltoall([t], bw_gbs) for t in large_tables] if large_tables else []

    total_threshold_us = 0.0
    total_threshold_launches = 0
    if small_result:
        total_threshold_us += small_result.total_time_us
        total_threshold_launches += small_result.num_launches
    for r in large_results:
        total_threshold_us += r.total_time_us
        total_threshold_launches += r.num_launches

    strategies["threshold_based"] = {
        "num_small_tables": len(small_tables),
        "num_large_tables": len(large_tables),
        "total_launches": total_threshold_launches,
        "total_time_us": round(total_threshold_us, 3),
    }

    return strategies


# ── Benchmark ────────────────────────────────────────────────────────

def generate_benchmark_scenarios() -> List[List[TableConfig]]:
    """Generate realistic recommendation model scenarios."""
    scenarios = []

    # Scenario 1: 16 small tables (typical ad CTR model)
    scenarios.append([
        TableConfig(f"emb_{i}", embedding_dim=64, num_tokens=512, first_dim=10000)
        for i in range(16)
    ])

    # Scenario 2: 32 medium tables
    scenarios.append([
        TableConfig(f"emb_{i}", embedding_dim=128, num_tokens=1024, first_dim=50000)
        for i in range(32)
    ])

    # Scenario 3: Mixed sizes (realistic)
    mixed = []
    for i in range(8):
        mixed.append(TableConfig(f"small_{i}", embedding_dim=32, num_tokens=256, first_dim=1000))
    for i in range(4):
        mixed.append(TableConfig(f"med_{i}", embedding_dim=128, num_tokens=1024, first_dim=10000))
    for i in range(2):
        mixed.append(TableConfig(f"large_{i}", embedding_dim=256, num_tokens=4096, first_dim=100000))
    scenarios.append(mixed)

    return scenarios


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Table AlltoAllv Aggregation Fusion Analysis"
    )
    parser.add_argument("--num-tables", type=int, default=16,
                        help="Number of embedding tables")
    parser.add_argument("--tokens-per-table", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--first-dim", type=int, default=10000)
    parser.add_argument("--benchmark-model", action="store_true",
                        help="Run benchmark model across scenarios")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")

    args = parser.parse_args()

    if args.benchmark_model:
        scenarios = generate_benchmark_scenarios()
        print("=" * 72)
        print("Multi-Table AlltoAllv Fusion: Scenario Analysis")
        print("=" * 72)

        all_results = []
        for i, tables in enumerate(scenarios):
            separate = model_separate_alltoall(tables)
            fused = model_fused_alltoall(tables)

            speedup = separate.total_time_us / max(fused.total_time_us, 1e-6)
            launch_reduction = (1 - fused.num_launches / separate.num_launches) * 100

            print(f"\nScenario {i+1}: {len(tables)} tables")
            print(f"  Separate: {separate.total_time_us:.1f} us "
                  f"({separate.num_launches} launches)")
            print(f"  Fused:    {fused.total_time_us:.1f} us "
                  f"({fused.num_launches} launches)")
            print(f"  Speedup:  {speedup:.1f}x")
            print(f"  Launch reduction: {launch_reduction:.0f}%")
            print(f"  BW separate: {separate.effective_bw_gbs:.1f} GB/s")
            print(f"  BW fused:    {fused.effective_bw_gbs:.1f} GB/s")

            strategies = compute_optimal_fusion_strategy(tables)
            all_results.append({
                "scenario": i + 1,
                "num_tables": len(tables),
                "separate": vars(separate),
                "fused": vars(fused),
                "speedup": round(speedup, 2),
                "launch_reduction_pct": round(launch_reduction, 1),
                "strategies": {k: v if isinstance(v, dict) else vars(v)
                              for k, v in strategies.items()},
            })

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
    else:
        tables = [
            TableConfig(f"emb_{i}", embedding_dim=args.embedding_dim,
                        num_tokens=args.tokens_per_table,
                        first_dim=args.first_dim)
            for i in range(args.num_tables)
        ]

        separate = model_separate_alltoall(tables)
        fused = model_fused_alltoall(tables)

        print(f"\n{'='*60}")
        print(f"Multi-Table AlltoAllv Fusion: {args.num_tables} tables")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'Separate':<15} {'Fused':<15}")
        print(f"  {'-'*55}")
        print(f"  {'Total time':<25} {separate.total_time_us:>10.1f} us  "
              f"{fused.total_time_us:>10.1f} us")
        print(f"  {'Num launches':<25} {separate.num_launches:>15}  "
              f"{fused.num_launches:>15}")
        print(f"  {'Control overhead':<25} {separate.control_overhead_us:>10.1f} us  "
              f"{fused.control_overhead_us:>10.1f} us")
        print(f"  {'Data transfer':<25} {separate.data_transfer_us:>10.1f} us  "
              f"{fused.data_transfer_us:>10.1f} us")
        print(f"  {'Effective BW':<25} {separate.effective_bw_gbs:>10.1f} GB/s "
              f"{fused.effective_bw_gbs:>10.1f} GB/s")
        speedup = separate.total_time_us / max(fused.total_time_us, 1e-6)
        print(f"\n  Speedup: {speedup:.1f}x\n")


if __name__ == "__main__":
    main()
