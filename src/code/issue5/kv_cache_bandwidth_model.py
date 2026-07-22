#!/usr/bin/env python3
"""
KV Cache "Store-instead-of-Compute" Bandwidth Critical Value Model
===================================================================
Analytical model for determining when loading KV Cache from remote storage
beats recomputing from scratch in LLM inference prefill.

Key insight: There exists a critical bandwidth threshold B_crit where:
  T_load(B) + T_recompute_unhit <= T_recompute_all

If actual bandwidth > B_crit, store-and-load wins.
If actual bandwidth < B_crit, recomputing is faster.

Usage:
  python kv_cache_bandwidth_model.py --hit-rate 0.3 --ctx-len 128K
  python kv_cache_bandwidth_model.py --rdma-analysis
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── Model Parameters ─────────────────────────────────────────────────

@dataclass
class KVCacheConfig:
    """KV Cache workload configuration."""
    # Model parameters
    num_layers: int = 80           # Number of transformer layers
    num_kv_heads: int = 8          # Number of KV heads
    head_dim: int = 128            # Dimension per head
    hidden_dim: int = 8192         # Hidden dimension

    # Workload parameters
    context_length: int = 131072   # Total context length (128K)
    prompt_length: int = 4096      # New prompt tokens to process

    # Cache parameters
    hit_rate: float = 0.3          # Fraction of KV cache already in remote storage

    # Hardware parameters
    gpu_tflops: float = 989.0      # GPU BF16 TFLOPS
    tcp_bandwidth_gbs: float = 2.5  # TCP bandwidth (observed: ~2.5 GB/s)
    rdma_bandwidth_gbs: float = 50.0  # RDMA bandwidth

    @property
    def kv_bytes_per_token(self) -> int:
        """KV cache size per token in bytes (BF16, K+V)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2

    @property
    def total_kv_cache_gb(self) -> float:
        """Total KV cache size for the full context."""
        return self.context_length * self.kv_bytes_per_token / 1e9

    @property
    def hit_kv_cache_gb(self) -> float:
        """Amount of KV cache that can be loaded (hit portion)."""
        return self.total_kv_cache_gb * self.hit_rate

    @property
    def unhit_kv_cache_gb(self) -> float:
        """Amount of KV cache that must be recomputed."""
        return self.total_kv_cache_gb * (1.0 - self.hit_rate)


# ── Bandwidth Critical Value Model ───────────────────────────────────

def compute_critical_bandwidth(
    config: KVCacheConfig,
) -> Dict:
    """
    Compute the critical bandwidth threshold.

    T_recompute_all = flops / GPU_TFLOPS
    T_load = KV_size / bandwidth
    T_recompute_unhit = (1 - hit_rate) * T_recompute_all

    Break-even: T_load + T_recompute_unhit = T_recompute_all
    => T_load = hit_rate * T_recompute_all
    => KV_size / B_crit = hit_rate * flops / GPU_TFLOPS
    => B_crit = KV_size * GPU_TFLOPS / (hit_rate * flops)

    Returns dict with all computed values.
    """
    # Recompute cost
    # Each token's KV requires: 2 (K+V) × num_layers × num_kv_heads × head_dim ops
    flops_per_token = (2 * config.num_layers * config.num_kv_heads *
                       config.head_dim * config.hidden_dim)
    total_recompute_flops = config.context_length * flops_per_token
    t_recompute_all = total_recompute_flops / (config.gpu_tflops * 1e12)

    # B_crit derivation from break-even
    kv_bytes = config.total_kv_cache_gb * 1e9
    b_crit = kv_bytes / (config.hit_rate * t_recompute_all)

    # Time components
    t_load_tcp = config.hit_kv_cache_gb / config.tcp_bandwidth_gbs
    t_load_rdma = config.hit_kv_cache_gb / config.rdma_bandwidth_gbs
    t_recompute_unhit = t_recompute_all * (1.0 - config.hit_rate)

    # Total time
    total_tcp = t_load_tcp + t_recompute_unhit
    total_rdma = t_load_rdma + t_recompute_unhit

    # QPM estimation (queries per minute)
    ttft_target = 0.5  # Target TTFT: 500ms
    qpm_tcp = 60.0 / max(total_tcp, ttft_target)
    qpm_rdma = 60.0 / max(total_rdma, ttft_target)

    return {
        # Critical bandwidth
        "critical_bandwidth_gbs": round(b_crit / 1e9, 2),
        "tcp_is_above_critical": config.tcp_bandwidth_gbs > b_crit / 1e9,
        "rdma_is_above_critical": config.rdma_bandwidth_gbs > b_crit / 1e9,

        # Timing breakdown
        "t_recompute_all_ms": round(t_recompute_all * 1000, 3),
        "t_load_tcp_ms": round(t_load_tcp * 1000, 3),
        "t_load_rdma_ms": round(t_load_rdma * 1000, 3),
        "t_recompute_unhit_ms": round(t_recompute_unhit * 1000, 3),
        "total_time_tcp_ms": round(total_tcp * 1000, 3),
        "total_time_rdma_ms": round(total_rdma * 1000, 3),

        # Throughput
        "qpm_tcp": round(qpm_tcp, 1),
        "qpm_rdma": round(qpm_rdma, 1),

        # KV cache stats
        "total_kv_cache_gb": round(config.total_kv_cache_gb, 2),
        "hit_kv_cache_gb": round(config.hit_kv_cache_gb, 2),
        "unhit_kv_cache_gb": round(config.unhit_kv_cache_gb, 2),

        # Break-even analysis
        "store_vs_recompute_tcp": "STORE" if total_tcp < t_recompute_all else "RECOMPUTE",
        "store_vs_recompute_rdma": "STORE" if total_rdma < t_recompute_all else "RECOMPUTE",
        "rdma_speedup_vs_tcp": round(total_tcp / max(total_rdma, 1e-6), 2),
        "qpm_gain_pct": round((qpm_rdma / max(qpm_tcp, 0.01) - 1.0) * 100, 1),
    }


def sweep_parameters(
    hit_rates: List[float],
    ctx_lengths: List[int],
    output_json: Optional[str] = None,
) -> List[Dict]:
    """
    Sweep across hit rates and context lengths to build decision surface.
    """
    results = []
    for hr in hit_rates:
        for ctx_len in ctx_lengths:
            config = KVCacheConfig(
                hit_rate=hr,
                context_length=ctx_len,
            )
            r = compute_critical_bandwidth(config)
            r["hit_rate"] = hr
            r["context_length"] = ctx_len
            results.append(r)

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)

    return results


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KV Cache Bandwidth Critical Value Model"
    )
    parser.add_argument("--hit-rate", type=float, default=0.3,
                        help="KV cache hit rate in remote storage")
    parser.add_argument("--ctx-len", type=str, default="128K",
                        help="Context length (e.g., 32K, 128K)")
    parser.add_argument("--rdma-analysis", action="store_true",
                        help="Run TCP vs RDMA comparison")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep hit rates and context lengths")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file")

    args = parser.parse_args()

    # Parse context length
    ctx_mult = {"K": 1024}
    ctx_str = args.ctx_len.upper()
    ctx_len = int(float(ctx_str[:-1]) * 1024) if ctx_str.endswith("K") else int(ctx_str)

    config = KVCacheConfig(
        hit_rate=args.hit_rate,
        context_length=ctx_len,
    )

    print("=" * 72)
    print("KV Cache 'Store-vs-Compute' Bandwidth Critical Value Analysis")
    print("=" * 72)
    print(f"  Context length:    {ctx_len:,} tokens")
    print(f"  Total KV cache:    {config.total_kv_cache_gb:.2f} GB")
    print(f"  Hit rate:          {args.hit_rate:.0%}")
    print(f"  Hit KV cache:      {config.hit_kv_cache_gb:.2f} GB")
    print()

    result = compute_critical_bandwidth(config)

    print("--- Timing Breakdown ---")
    print(f"  Full recompute:    {result['t_recompute_all_ms']:>8.2f} ms")
    print(f"  TCP load:          {result['t_load_tcp_ms']:>8.2f} ms")
    print(f"  RDMA load:         {result['t_load_rdma_ms']:>8.2f} ms")
    print(f"  Unhit recompute:   {result['t_recompute_unhit_ms']:>8.2f} ms")
    print()

    print("--- Total Time ---")
    print(f"  TCP total:         {result['total_time_tcp_ms']:>8.2f} ms -> "
          f"{result['store_vs_recompute_tcp']}")
    print(f"  RDMA total:        {result['total_time_rdma_ms']:>8.2f} ms -> "
          f"{result['store_vs_recompute_rdma']}")
    print()

    print(f"  Critical bandwidth: {result['critical_bandwidth_gbs']:.2f} GB/s")
    print(f"  TCP ({config.tcp_bandwidth_gbs} GB/s): "
          f"{'ABOVE' if result['tcp_is_above_critical'] else 'BELOW'} critical")
    print(f"  RDMA ({config.rdma_bandwidth_gbs} GB/s): "
          f"{'ABOVE' if result['rdma_is_above_critical'] else 'BELOW'} critical")
    print()

    print("--- Throughput ---")
    print(f"  QPM (TCP):  {result['qpm_tcp']:.0f}")
    print(f"  QPM (RDMA): {result['qpm_rdma']:.0f}  (+{result['qpm_gain_pct']:.0f}%)")
    print()

    if args.sweep:
        hit_rates = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
        ctx_lengths = [8192, 32768, 65536, 131072, 262144]
        sweep_results = sweep_parameters(hit_rates, ctx_lengths, args.output)
        print(f"Sweep: {len(sweep_results)} configurations analyzed")
        print("\nDecision matrix (STORE=green, RECOMPUTE=red):")
        print(f"{'Hit Rate':<10}", end="")
        for cl in ctx_lengths:
            print(f"{cl//1024}K".rjust(12), end="")
        print()
        for hr in hit_rates:
            print(f"{hr:<10.0%}", end="")
            for cl in ctx_lengths:
                r = [x for x in sweep_results
                     if x["hit_rate"] == hr and x["context_length"] == cl][0]
                marker = "S" if r["store_vs_recompute_rdma"] == "STORE" else "R"
                print(f"{marker:>12}", end="")
            print()

    if args.output and not args.sweep:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
