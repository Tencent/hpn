"""Run independent NCCL jobs and compare their output bytes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from core import compare_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"), default="all_reduce")
    parser.add_argument("--elements", type=int, default=1 << 18)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--algo", help="NCCL_ALGO value, e.g. Ring or Tree")
    parser.add_argument("--proto", help="NCCL_PROTO value, e.g. Simple, LL, or LL128")
    parser.add_argument("--topo-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic-output"))
    parser.add_argument("--keep-payloads", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 2 or args.nproc_per_node < 2:
        raise ValueError("--runs and --nproc-per-node must both be at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("worker.py")
    captures: list[Path] = []
    env = os.environ.copy()
    for key, value in (
        ("NCCL_ALGO", args.algo),
        ("NCCL_PROTO", args.proto),
        ("NCCL_TOPO_FILE", str(args.topo_file.resolve()) if args.topo_file else None),
    ):
        if value:
            env[key] = value
        else:
            env.pop(key, None)

    for run in range(args.runs):
        capture = args.output_dir / f"run-{run:02d}.json"
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={args.nproc_per_node}",
            str(worker),
            "--output",
            str(capture),
            "--op",
            args.op,
            "--elements",
            str(args.elements),
            "--calls",
            str(args.calls),
            "--dtype",
            args.dtype,
            "--seed",
            str(args.seed),
        ]
        print(f"[run {run + 1}/{args.runs}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, env=env, check=True)
        captures.append(capture)

    report = compare_runs(captures)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["first_divergence"], indent=2))
    print(f"bitwise_identical={report['bitwise_identical']} report={report_path}")
    if not args.keep_payloads:
        for capture in captures:
            capture.unlink()
    return 0 if report["bitwise_identical"] else 2


if __name__ == "__main__":
    sys.exit(main())
