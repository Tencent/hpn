"""Execute a controlled NCCL algorithm/protocol matrix and summarize it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"), default="all_reduce")
    parser.add_argument("--elements", type=int, default=1 << 18)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--algos", default="default,Ring,Tree")
    parser.add_argument("--protos", default="default,Simple,LL")
    parser.add_argument("--output-dir", type=Path, default=Path("sweep-output"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    driver = Path(__file__).with_name("diagnose.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for algo in args.algos.split(","):
        for proto in args.protos.split(","):
            # "default + explicit" duplicates an explicit override because the
            # unset dimension remains under NCCL's automatic selection.
            slug = f"{algo.lower()}-{proto.lower()}"
            output = args.output_dir / slug
            cmd = [
                sys.executable,
                str(driver),
                "--nproc-per-node",
                str(args.nproc_per_node),
                "--runs",
                str(args.runs),
                "--op",
                args.op,
                "--elements",
                str(args.elements),
                "--calls",
                str(args.calls),
                "--dtype",
                args.dtype,
                "--output-dir",
                str(output),
            ]
            if algo != "default":
                cmd.extend(("--algo", algo))
            if proto != "default":
                cmd.extend(("--proto", proto))
            print(f"\n=== ALGO={algo} PROTO={proto} ===", flush=True)
            result = subprocess.run(cmd, env=os.environ.copy())
            report_path = output / "report.json"
            row: dict[str, object] = {
                "algo": algo,
                "proto": proto,
                "exit_code": result.returncode,
                "report": str(report_path),
            }
            if report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                row["bitwise_identical"] = report["bitwise_identical"]
                row["first_divergence"] = report["first_divergence"]
            else:
                row["error"] = "launch failed; inspect console/NCCL logs"
            rows.append(row)

    summary = {
        "experiment": {
            "op": args.op,
            "elements": args.elements,
            "calls": args.calls,
            "runs": args.runs,
            "world_size": args.nproc_per_node,
            "dtype": args.dtype,
        },
        "results": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary={summary_path}")
    return 1 if any("error" in row for row in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
