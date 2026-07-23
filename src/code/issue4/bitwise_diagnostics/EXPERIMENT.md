# Four-GPU reproducibility experiment

Date: 2026-07-23

The machine-readable summary is
[`results/autodl-4x3090.json`](results/autodl-4x3090.json).

## Environment

- 4 × NVIDIA GeForce RTX 3090 24 GiB, driver 580.105.08
- PyTorch 2.8.0+cu128, CUDA 12.8, NCCL 2.27.3
- GPU 0↔1 and GPU 2↔3: `PXB`; traffic between the pairs: `SYS`
- two NUMA nodes; no NVLink reported by `nvidia-smi topo -m`
- fixed seed 2026 and rank-distinct float32 inputs

Each reported run is a fresh four-rank `torchrun --standalone` launch. This
matters: repeated calls inside one communicator do not test communicator
initialization or run-to-run selection.

## Same-configuration results

At 1 MiB input per rank, AllReduce covered nine
`NCCL_ALGO × NCCL_PROTO` configurations:

| Algorithm | Protocols | Independent runs | Calls/run | Result |
|---|---|---:|---:|---|
| automatic | automatic, Simple, LL | 5 each | 20 | bitwise identical |
| Ring | automatic, Simple, LL | 5 each | 20 | bitwise identical |
| Tree | automatic, Simple, LL | 5 each | 20 | bitwise identical |

Reduce-Scatter used the same input size and repetition count:

| Algorithm | Protocols | Result |
|---|---|---|
| automatic | automatic, Simple, LL | bitwise identical |
| Ring | automatic, Simple, LL | bitwise identical |
| Tree | automatic, Simple, LL | rejected with `ncclInvalidUsage` |

Tree is therefore not presented as a Reduce-Scatter mitigation on this NCCL
version. Importantly, the runner did not silently fall back to Ring.

No same-configuration run-to-run divergence was observed. This is a bounded
negative result for this exact software and PCIe topology, not a universal
claim that these algorithms are deterministic.

## Granularity

NCCL automatic selection was also checked across message sizes:

| Bytes/rank | Independent runs | Calls/run | Result |
|---:|---:|---:|---|
| 1 KiB | 5 | 50 | bitwise identical |
| 64 KiB | 5 | 30 | bitwise identical |
| 1 MiB | 5 | 20 | bitwise identical |
| 16 MiB | 5 | 3 | bitwise identical |

The call count is lower for large messages because the diagnostic retains raw
bytes until comparison. Every row still compares independent launches with
the same call count within the row.

## Reproducible algorithm-change case

To verify localization and demonstrate the numerical consequence of reduction
order, one factor was changed from Ring to Tree while seed, rank inputs,
hardware, dtype, message size, and call sequence stayed fixed.

The first difference appeared at call 0, rank 0, byte 24 (float32 element 6).
For that 1 MiB output:

- 94,656 bytes / 192,767 bits changed;
- maximum absolute error was `9.5367431640625e-07`;
- maximum reported ULP distance was 49,152;
- Ring SHA-256:
  `c5797c4ea5b9d3a387fab7d4e3a6405fa60f3397d19a050070e896433b99d513`;
- Tree SHA-256:
  `3f3470571b53caf023710336ca21a493c7a27e27e40871280a5c85a44d44a991`.

This is a controlled configuration-change case, **not** evidence of
nondeterminism within either fixed configuration. It demonstrates why
algorithm/topology selection must remain stable when bitwise continuity
between jobs is required.

## Conclusions by acceptance dimension

1. **Granularity:** no run-to-run difference was observed from 1 KiB through
   16 MiB under automatic selection. Larger captures cost proportionally more
   memory; sampling call count is an explicit experimental trade-off.
2. **Hardware:** results apply to a four-GPU, dual-NUMA PCIe host. The `SYS`
   boundary between GPU pairs is materially different from NVLink/NVSwitch;
   NVLS was therefore not tested or recommended.
3. **Software flow:** fresh process groups, fixed rank mapping, fixed inputs,
   and explicit algorithm/protocol settings produced stable bytes. Switching
   Ring to Tree immediately changed results. For reproducible jobs, pin the
   tested software stack and selection controls, but only use algorithms that
   support the target collective.

The evidence supports fixed Ring (including Simple or LL on this host) as a
measured reproducible option for both tested collectives. This is deliberately
scoped to the recorded environment rather than framed as a universal NCCL
guarantee.
