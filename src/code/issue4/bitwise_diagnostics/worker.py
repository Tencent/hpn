"""One independent distributed NCCL capture. Launch with torchrun."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--op", choices=("all_reduce", "reduce_scatter"), default="all_reduce")
    parser.add_argument("--elements", type=int, default=1 << 18)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def nccl_version_string() -> str:
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return ".".join(map(str, version))
    return str(version)


def main() -> None:
    args = parse_args()
    if args.elements <= 0 or args.calls <= 0:
        raise ValueError("--elements and --calls must be positive")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if args.op == "reduce_scatter" and args.elements % world_size:
        raise ValueError("--elements must be divisible by world size for reduce_scatter")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    source = torch.randn(args.elements, dtype=dtype, generator=generator).cuda(local_rank)
    captures: list[dict[str, object]] = []

    for call in range(args.calls):
        # Preserve identical per-rank inputs across independent launches while
        # changing data between calls so the first divergent call is meaningful.
        input_tensor = source + call * torch.finfo(dtype).eps
        if args.op == "all_reduce":
            output = input_tensor.clone()
            dist.all_reduce(output)
        else:
            output = torch.empty(args.elements // world_size, dtype=dtype, device=local_rank)
            dist.reduce_scatter_tensor(output, input_tensor.contiguous())
        torch.cuda.synchronize()
        raw = output.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        captures.append(
            {
                "call": call,
                "rank": rank,
                "dtype": str(dtype),
                "shape": list(output.shape),
                "payload_hex": raw.hex(),
            }
        )

    gathered: list[list[dict[str, object]] | None] | None = (
        [None] * world_size if rank == 0 else None
    )
    dist.gather_object(captures, gathered, dst=0)
    if rank == 0:
        metadata = {
            "op": args.op,
            "dtype": str(dtype),
            "elements": args.elements,
            "calls": args.calls,
            "world_size": world_size,
            "seed": args.seed,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": nccl_version_string(),
            "gpu": torch.cuda.get_device_name(local_rank),
            "host": platform.node(),
            "nccl_env": {
                key: os.environ[key]
                for key in ("NCCL_ALGO", "NCCL_PROTO", "NCCL_TOPO_FILE")
                if key in os.environ
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "metadata": metadata,
                    "captures": [item for rank_items in gathered or [] for item in rank_items or []],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
