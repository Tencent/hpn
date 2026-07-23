"""Pure-Python comparison and reporting primitives for NCCL run captures."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TensorCapture:
    call: int
    rank: int
    dtype: str
    shape: list[int]
    payload_hex: str

    @property
    def payload(self) -> bytes:
        return bytes.fromhex(self.payload_hex)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class Difference:
    run: int
    call: int
    rank: int
    first_byte: int
    first_element: int
    changed_bytes: int
    changed_bits: int
    max_abs_error: float | None
    max_ulp_error: int | None
    baseline_sha256: str
    candidate_sha256: str


def _float_format(dtype: str) -> tuple[str, str, int] | None:
    return {
        "torch.float16": ("e", "H", 2),
        "torch.float32": ("f", "I", 4),
        "torch.float64": ("d", "Q", 8),
    }.get(dtype)


def _ordered_int(bits: int, width: int) -> int:
    """Map IEEE sign-magnitude bit patterns to monotonically ordered integers."""
    sign = 1 << (width * 8 - 1)
    mask = (1 << (width * 8)) - 1
    return (~bits & mask) if bits & sign else (bits | sign)


def compare_capture(
    baseline: TensorCapture, candidate: TensorCapture, run: int
) -> Difference | None:
    if (baseline.dtype, baseline.shape) != (candidate.dtype, candidate.shape):
        raise ValueError(
            f"capture metadata changed at call={baseline.call}, rank={baseline.rank}"
        )
    left, right = baseline.payload, candidate.payload
    if len(left) != len(right):
        raise ValueError("capture payload lengths differ")
    if left == right:
        return None

    xor = bytes(a ^ b for a, b in zip(left, right))
    first_byte = next(i for i, byte in enumerate(xor) if byte)
    changed_bytes = sum(bool(byte) for byte in xor)
    changed_bits = sum(byte.bit_count() for byte in xor)
    fmt = _float_format(baseline.dtype)
    max_abs: float | None = None
    max_ulp: int | None = None
    element_size = fmt[2] if fmt else 1

    if fmt:
        float_code, int_code, width = fmt
        count = len(left) // width
        left_values = struct.unpack(f"<{count}{float_code}", left)
        right_values = struct.unpack(f"<{count}{float_code}", right)
        left_bits = struct.unpack(f"<{count}{int_code}", left)
        right_bits = struct.unpack(f"<{count}{int_code}", right)
        abs_errors = [
            abs(a - b)
            for a, b in zip(left_values, right_values)
            if not (math.isnan(a) and math.isnan(b))
        ]
        max_abs = max(abs_errors, default=0.0)
        max_ulp = max(
            abs(_ordered_int(a, width) - _ordered_int(b, width))
            for a, b in zip(left_bits, right_bits)
        )

    return Difference(
        run=run,
        call=baseline.call,
        rank=baseline.rank,
        first_byte=first_byte,
        first_element=first_byte // element_size,
        changed_bytes=changed_bytes,
        changed_bits=changed_bits,
        max_abs_error=max_abs,
        max_ulp_error=max_ulp,
        baseline_sha256=baseline.sha256,
        candidate_sha256=candidate.sha256,
    )


def load_capture(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported capture schema in {path}")
    return data


def compare_runs(run_paths: Iterable[Path]) -> dict[str, Any]:
    paths = list(run_paths)
    if len(paths) < 2:
        raise ValueError("at least two independent runs are required")
    runs = [load_capture(path) for path in paths]
    keys = ("op", "dtype", "elements", "calls", "world_size", "seed")
    baseline_meta = {key: runs[0]["metadata"][key] for key in keys}
    for index, run in enumerate(runs[1:], 1):
        candidate_meta = {key: run["metadata"][key] for key in keys}
        if candidate_meta != baseline_meta:
            raise ValueError(f"run {index} is not comparable to the baseline")

    baseline = {
        (item["call"], item["rank"]): TensorCapture(**item)
        for item in runs[0]["captures"]
    }
    differences: list[Difference] = []
    for run_index, run in enumerate(runs[1:], 1):
        candidate = {
            (item["call"], item["rank"]): TensorCapture(**item)
            for item in run["captures"]
        }
        if candidate.keys() != baseline.keys():
            raise ValueError(f"run {run_index} has an incomplete capture set")
        for key in sorted(baseline):
            difference = compare_capture(baseline[key], candidate[key], run_index)
            if difference:
                differences.append(difference)

    first = min(differences, key=lambda item: (item.call, item.run, item.rank), default=None)
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": baseline_meta,
        "run_files": [str(path) for path in paths],
        "bitwise_identical": not differences,
        "first_divergence": asdict(first) if first else None,
        "differences": [asdict(item) for item in differences],
    }
