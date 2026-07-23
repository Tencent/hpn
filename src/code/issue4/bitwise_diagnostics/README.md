# NCCL bitwise reproducibility diagnostics

This tool launches the same collective workload in **separate `torchrun`
process groups**, captures every output as raw bytes, and compares all ranks
and calls against the first run. A JSON report identifies the first divergent
call, rank, byte/element offset, changed bits, absolute error, and ULP error.

Unlike an arithmetic simulation, this exercises the installed PyTorch, NCCL,
CUDA runtime, topology discovery, and actual communication path.

## Requirements

- Linux, Python 3.10+, PyTorch with NCCL, and at least two CUDA GPUs
- All tested ranks must see the same GPUs in every independent run
- For multi-node experiments, invoke `worker.py` with the site's normal
  rendezvous command and compare its capture files with `core.compare_runs`

No claim of determinism is inferred from an algorithm name. The report only
describes the measured hardware/software configuration.

See [`EXPERIMENT.md`](EXPERIMENT.md) for measured four-GPU results covering
algorithm/protocol selection, message granularity, both collectives, and a
controlled Ring-versus-Tree difference.

## Quick start

```bash
cd src/code/issue4/bitwise_diagnostics

# NCCL automatic selection
python diagnose.py --nproc-per-node 8 --runs 5 \
  --op all_reduce --elements 1048576 --calls 20 \
  --output-dir results/default

# Pin one supported algorithm/protocol for an A/B comparison
python diagnose.py --nproc-per-node 8 --runs 5 \
  --algo Ring --proto Simple --op all_reduce \
  --elements 1048576 --calls 20 --output-dir results/ring-simple

# Unified A/B matrix; unsupported combinations are recorded, not hidden
python sweep.py --nproc-per-node 8 --runs 5 \
  --algos default,Ring,Tree --protos default,Simple,LL
```

Exit status is `0` for bitwise-identical runs, `2` when a divergence is
detected, and non-zero on invalid configuration or launch failure. Use
`--keep-payloads` when forensic inspection is needed; otherwise multi-MB raw
captures are deleted after `report.json` is written.

Existing captures can also be compared offline (including captures copied
from different nodes):

```bash
python core.py run-a.json run-b.json --output comparison.json
```

Run CPU-only unit tests with:

```bash
python -m unittest -v test_core.py
```

## Controlled experiment matrix

Change one factor at a time and retain `report.json` for each row:

1. **Message granularity:** 1 KiB, 64 KiB, 1 MiB, 16 MiB, 128 MiB. Since
   raw captures scale as `elements × dtype bytes × calls × ranks × runs`,
   reduce `--calls` for the largest cases and keep the value identical across
   compared configurations.
2. **Collective:** `all_reduce` and `reduce_scatter`.
3. **Selection:** NCCL default, then supported `NCCL_ALGO` values with
   `Simple`, `LL`, and (only on supported platforms) `LL128`.
4. **Resources:** record GPU model/count, NVLink/NVSwitch, NIC, node count,
   PyTorch/CUDA/NCCL versions, and topology. Worker metadata records the
   software versions and effective experiment overrides automatically.
5. **Stability:** at least five independent launches and 50 calls per launch.

Do not force unsupported combinations. NCCL 2.24+ fails on invalid algorithm
tokens, and NVIDIA warns that forcing LL128 on unsupported platforms can cause
data corruption. `NCCL_ALGO` and `NCCL_PROTO` are diagnostic controls, not
universal production recommendations.

## Interpreting results across the three acceptance dimensions

- **Granularity:** small messages emphasize launch/protocol behavior; large
  messages exercise more chunks/channels and expose more reduction sites.
  Compare divergence rate and first-call position alongside latency measured
  by a dedicated benchmark such as `nccl-tests`.
- **Hardware resources:** rank count and topology determine valid algorithms
  and reduction paths. A result on NVSwitch must not be generalized to PCIe or
  multi-node fabrics. A pinned `NCCL_TOPO_FILE` is useful only when it
  accurately represents the tested system.
- **Software flow:** fresh process groups test run-to-run reproducibility,
  which repeated calls inside one communicator cannot establish. Pin package
  versions and seeds; serialize collectives in the same order on every rank.
  PyTorch deterministic-algorithm settings and
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` matter for surrounding compute, but they do
  not constitute evidence that a collective is bitwise reproducible.

## Reproducible non-determinism case

The tool deliberately does not fabricate nondeterminism. To document a case:

1. run the default selection matrix on the target cluster;
2. retain the first report with `bitwise_identical=false`;
3. repeat with one variable pinned at a time;
4. report a mitigation only if repeated measurements turn identical;
5. attach NCCL `INFO` logs to show the selected algorithm/topology.

This separates observed evidence from assumptions about Ring, Tree, PAT, or
NVLS internals and avoids recommending undocumented environment variables.

## References

- NVIDIA NCCL environment variables:
  https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- PyTorch reproducibility:
  https://docs.pytorch.org/docs/stable/notes/randomness.html
- PyTorch deterministic algorithms:
  https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
