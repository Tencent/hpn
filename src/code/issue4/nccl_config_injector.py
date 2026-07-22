#!/usr/bin/env python3
"""
NCCL Configuration Injector for Determinism Testing
====================================================
Helper to programmatically set NCCL environment variables and
inject different algorithm/protocol/topology configurations
for systematic determinism testing.

Supports:
- Setting NCCL_ALGO, NCCL_PROTO, NCCL_DETERMINISTIC
- Topology pinning via NCCL_TOPO_FILE
- Debug output via NCCL_DEBUG
- Integration with torch.distributed and nccl-tests
"""

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── NCCL Configuration ───────────────────────────────────────────────

@dataclass
class NCCLConfig:
    """Complete NCCL configuration for a test run."""
    algo: Optional[str] = None          # NCCL_ALGO: Ring, Tree, NVLS, PAT
    proto: Optional[str] = None         # NCCL_PROTO: Simple, LL, LL128
    deterministic: bool = True          # NCCL_DETERMINISTIC
    nvls_enable: Optional[bool] = None  # NCCL_NVLS_ENABLE
    min_ctas: Optional[int] = None      # NCCL_MIN_CTAS
    max_ctas: Optional[int] = None      # NCCL_MAX_CTAS
    topo_file: Optional[str] = None     # NCCL_TOPO_FILE
    debug_level: str = "WARN"           # NCCL_DEBUG
    debug_file: Optional[str] = None    # NCCL_DEBUG_FILE
    cublas_workspace_config: str = ":4096:8"  # CUBLAS_WORKSPACE_CONFIG

    def to_env(self) -> Dict[str, str]:
        """Convert config to environment variable dict."""
        env = {}
        if self.algo:
            env["NCCL_ALGO"] = self.algo
        if self.proto:
            env["NCCL_PROTO"] = self.proto
        env["NCCL_DETERMINISTIC"] = "1" if self.deterministic else "0"
        if self.nvls_enable is not None:
            env["NCCL_NVLS_ENABLE"] = "1" if self.nvls_enable else "0"
        if self.min_ctas:
            env["NCCL_MIN_CTAS"] = str(self.min_ctas)
        if self.max_ctas:
            env["NCCL_MAX_CTAS"] = str(self.max_ctas)
        if self.topo_file:
            env["NCCL_TOPO_FILE"] = self.topo_file
        env["NCCL_DEBUG"] = self.debug_level
        if self.debug_file:
            env["NCCL_DEBUG_FILE"] = self.debug_file
        env["CUBLAS_WORKSPACE_CONFIG"] = self.cublas_workspace_config
        return env

    @classmethod
    def deterministic_ring(cls) -> "NCCLConfig":
        """Deterministic ring-based config (best for internode)."""
        return cls(algo="Ring", proto="Simple", deterministic=True)

    @classmethod
    def deterministic_nvls(cls) -> "NCCLConfig":
        """Deterministic NVLS-based config (best for intranode with NVSwitch)."""
        return cls(algo="NVLS", nvls_enable=True, deterministic=True)

    @classmethod
    def deterministic_tree(cls) -> "NCCLConfig":
        """Tree-based config with pinned topology."""
        return cls(algo="Tree", proto="Simple", deterministic=True)

    @classmethod
    def default_nccl(cls) -> "NCCLConfig":
        """Default NCCL behavior (no constraints)."""
        return cls(algo=None, proto=None, deterministic=False)


# ── Config Sweep Generator ───────────────────────────────────────────

def generate_config_matrix() -> List[NCCLConfig]:
    """Generate all algorithm × protocol × determinism combinations."""
    configs = []
    algos = ["Ring", "Tree", "NVLS", "PAT", None]  # None = NCCL default
    protos = ["Simple", "LL", "LL128", None]

    for algo in algos:
        for proto in protos:
            for det in [True, False]:
                configs.append(NCCLConfig(
                    algo=algo,
                    proto=proto,
                    deterministic=det,
                ))

    return configs


def generate_focused_configs() -> List[NCCLConfig]:
    """
    Generate focused configs for determinism testing.
    Skips combinations known to be non-deterministic to avoid combinatorial explosion.
    """
    return [
        # Known deterministic combinations
        NCCLConfig.deterministic_ring(),
        NCCLConfig.deterministic_nvls(),
        NCCLConfig.deterministic_tree(),
        # Potentially non-deterministic
        NCCLConfig(algo="PAT", deterministic=False),
        NCCLConfig(algo="Tree", proto="LL128", deterministic=False),
        # Default (no constraints)
        NCCLConfig.default_nccl(),
        # For comparison
        NCCLConfig(algo="Ring", proto="LL", deterministic=True),
        NCCLConfig(algo="Tree", proto="LL", deterministic=True),
    ]


# ── Topology Pinning ─────────────────────────────────────────────────

def generate_deterministic_topo_file(
    num_gpus: int = 8,
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a minimal topology XML that pins GPU connections
    to ensure consistent routing across runs.
    """
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".xml", prefix="nccl_topo_")
        os.close(fd)

    xml = f'''<system version="1">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64" vendor="GenuineIntel" family="6" model="85">
    {chr(10).join(f'<pci busid="0000:{i:02x}:00.0" class="0x030200" vendor="0x10de" device="0x1db4" link_speed="16 GT/s" link_width="16"/>' for i in range(num_gpus))}
    {chr(10).join(f'<nic busid="0000:{i+32:02x}:00.0" link_speed="200000"/>' for i in range(8))}
  </cpu>
</system>'''

    with open(output_path, 'w') as f:
        f.write(xml)

    return output_path


# ── Environment Context Manager ──────────────────────────────────────

@contextmanager
def nccl_env_context(config: NCCLConfig):
    """Temporarily set NCCL environment variables."""
    old_env = {}
    new_env = config.to_env()

    for key, value in new_env.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        yield new_env
    finally:
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


# ── Integration Helpers ──────────────────────────────────────────────

def run_nccl_test(
    config: NCCLConfig,
    test_binary: str = "all_reduce_perf",
    min_bytes: str = "1K",
    max_bytes: str = "128M",
    num_gpus: int = 8,
) -> Tuple[int, str, str]:
    """
    Run nccl-tests with a specific NCCL configuration.
    Returns (returncode, stdout, stderr).
    """
    env = config.to_env()
    env.update(os.environ)

    cmd = [
        test_binary,
        "-b", min_bytes,
        "-e", max_bytes,
        "-g", str(num_gpus),
        "-n", "5",           # 5 iterations
        "-w", "2",           # 2 warmup
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )

    return proc.returncode, proc.stdout, proc.stderr


# ── Main ─────────────────────────────────────────────────────────────

def main():
    """Example usage of the NCCL config injector."""
    import argparse

    parser = argparse.ArgumentParser(
        description="NCCL Configuration Injector for Determinism Testing"
    )
    parser.add_argument("--list-configs", action="store_true",
                        help="List all focused test configurations")
    parser.add_argument("--generate-topo", action="store_true",
                        help="Generate deterministic topology XML")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for topology XML")

    args = parser.parse_args()

    if args.list_configs:
        configs = generate_focused_configs()
        print("Focused NCCL Configurations for Determinism Testing:")
        print("-" * 72)
        for i, cfg in enumerate(configs):
            env = cfg.to_env()
            print(f"\nConfig #{i+1}:")
            for k, v in sorted(env.items()):
                print(f"  {k}={v}")

    if args.generate_topo:
        path = generate_deterministic_topo_file(output_path=args.output)
        print(f"Topology XML written to: {path}")


if __name__ == "__main__":
    main()
