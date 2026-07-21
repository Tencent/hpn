#!/bin/bash
# =============================================================================
# Issue5 TCP/RDMA 基准 (收敛协议切换, 验收2 实现部分)
# -----------------------------------------------------------------------------
# 把 L1(transfer 原语) / L2(真实 KV 张量) / L4(TTFT-QPM replay) 的协议切换收敛到
# 一个入口: `TRANSPORT=tcp|rdma bash run_benchmark.sh <layer>`。
#
# 关键实现事实(见 configs/{tcp,rdma}.yaml、rdma/README.md):
#   - TCP↔RDMA 的**唯一传输改动** = Mooncake protocol + client device_name。
#   - RDMA client 必须显式 MC_DEVICE=mlx5_3 (prefill 的 eth1); 空会 No available RNIC。
#   - store 端需先用同一 MC_PROTOCOL 起 (MC_PROTOCOL=rdma bash start_kvstore.sh)。
#   - L4 纯本地重放(吃 L1-L3 CSV), 与协议无关, 一次跑出三传输对比, 无需切协议。
#
# 用法 (在 prefill pod 内, 或本地对已产出的 CSV 只跑 L4/汇总):
#   TRANSPORT=tcp  bash scripts/run_benchmark.sh l1        # L1 原语微基准
#   TRANSPORT=rdma MC_DEVICE=mlx5_3 bash scripts/run_benchmark.sh l1
#   TRANSPORT=rdma MC_DEVICE=mlx5_3 bash scripts/run_benchmark.sh l2   # L2 真实 KV
#   bash scripts/run_benchmark.sh l4                       # L4 replay (含中/高命中)
#   bash scripts/run_benchmark.sh compare                  # 只读 CSV 出 TCP/RDMA 对比表
#   TRANSPORT=rdma MC_DEVICE=mlx5_3 bash scripts/run_benchmark.sh all
#
# 环境变量:
#   TRANSPORT   tcp(默认) | rdma          —— 决定 L1/L2 走哪个协议
#   MC_DEVICE   RDMA NIC 名, rdma 必填(prefill=mlx5_3); tcp 留空
#   KV_IP       store 宿主机 IP (默认 192.0.2.10)
#   HOST_IP     本 pod IP (默认 hostname -i)
#   OUTDIR      CSV 输出目录 (默认 results/raw)
# =============================================================================
set -euo pipefail

LAYER="${1:-all}"
export TRANSPORT="${TRANSPORT:-tcp}"
export MC_DEVICE="${MC_DEVICE:-}"
export KV_IP="${KV_IP:-192.0.2.10}"
export HOST_IP="${HOST_IP:-$(hostname -i 2>/dev/null | awk '{print $1}')}"

# 定位仓库根 (scripts/ 的上一级)
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SELF/.." && pwd)"
OUTDIR="${OUTDIR:-$ROOT/results/raw}"
BM="$ROOT/benchmark"
mkdir -p "$OUTDIR"

# rdma 协议下强制要求 device (否则 client 自动发现失败 ret=-600)
if [[ "$TRANSPORT" == "rdma" && -z "$MC_DEVICE" ]]; then
  echo "[run_benchmark] TRANSPORT=rdma 但 MC_DEVICE 为空 —— prefill 端必须显式 MC_DEVICE=mlx5_3" >&2
  echo "                (client device_name='' 自动发现会 No available RNIC ret=-600, 见 rdma/README.md)" >&2
  exit 2
fi

TAG="$TRANSPORT"
echo "[run_benchmark] layer=$LAYER transport=$TRANSPORT device='${MC_DEVICE}' KV_IP=$KV_IP HOST_IP=$HOST_IP outdir=$OUTDIR"

run_l1() {
  echo "== L1 transfer 原语微基准 ($TRANSPORT) =="
  python3 "$BM/benchmark_transfer.py" \
    --protocol "$TRANSPORT" --device "$MC_DEVICE" \
    --out "$OUTDIR/transfer_${TAG}.csv"
}

run_l2() {
  echo "== L2 真实 KV 张量微基准 ($TRANSPORT) =="
  python3 "$BM/benchmark_kv_tensor.py" \
    --protocol "$TRANSPORT" --device "$MC_DEVICE" \
    --paths bytes,zerocopy \
    --out "$OUTDIR/kv_tensor_${TAG}.csv"
}

run_l4() {
  echo "== L4 TTFT/QPM replay (协议无关, 吃 L1-L3 CSV) =="
  # 中等命中 (算力主导): h=0.5 phit=0.5
  python3 "$BM/benchmark_concurrency.py" --context 8k,16k,24k \
    --hit-prefix 0.5 --phit 0.5 --slo-factor 2.5 --out-prefix "$OUTDIR/conc"
  # 高命中 (retrieve 主导): h=0.95 phit=0.8
  python3 "$BM/benchmark_concurrency.py" --context 8k,16k,24k \
    --hit-prefix 0.95 --phit 0.8 --slo-factor 2.5 --out-prefix "$OUTDIR/conc_hihit"
}

# 只读已产出的 CSV 出 TCP vs RDMA 对比表 (不重跑, 便于本地汇总)
run_compare() {
  echo "== TCP vs RDMA 对比 (读 $OUTDIR/*.csv) =="
  python3 - "$OUTDIR" <<'PY'
import csv, os, statistics, sys
outdir = sys.argv[1]
def med(path, pred, col):
    if not os.path.exists(path): return None
    vals=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if pred(r):
                try: vals.append(float(r[col]))
                except (ValueError, KeyError): pass
    return statistics.median(vals) if vals else None

print("\n[L1 原语] put/get 有效带宽峰值 (GB/s):")
for tp in ("tcp","rdma"):
    p=f"{outdir}/transfer_{tp}.csv"
    if not os.path.exists(p): print(f"  {tp}: (无 {p})"); continue
    puts=med(p, lambda r:r.get("op")=="put" and r.get("success")=="1", "bandwidth_gbps")
    gets=med(p, lambda r:r.get("op")=="get" and r.get("success")=="1", "bandwidth_gbps")
    print(f"  {tp:4}: put~{puts} get~{gets}")

print("\n[L2 真实 KV] zerocopy get 中位带宽 (GB/s):")
z={}
for tp in ("tcp","rdma"):
    p=f"{outdir}/kv_tensor_{tp}.csv"
    z[tp]=med(p, lambda r:r.get("path")=="zerocopy" and r.get("op")=="get" and r.get("success")=="1",
              "effective_bandwidth_gbps")
    print(f"  {tp:4}: zerocopy_get~{z[tp]}")
if z.get("tcp") and z.get("rdma"):
    print(f"  => RDMA-zc / TCP-zc = {z['rdma']/z['tcp']:.1f}x")

print("\n[L4 QPM] 各 regime × 传输 (读 conc_summary / conc_hihit_summary):")
for tag,label in (("conc","中等命中 h=.5"),("conc_hihit","高命中 h=.95")):
    p=f"{outdir}/{tag}_summary.csv"
    if not os.path.exists(p): print(f"  {label}: (无 {p})"); continue
    byN={}
    with open(p) as f:
        for r in csv.DictReader(f):
            byN.setdefault(int(r["context_length"]),{})[r["transport"]]=float(r["qpm"])
    print(f"  {label}:")
    for N,d in sorted(byN.items()):
        rt = d["rdma"]/d["tcp"] if d.get("tcp") else float("nan")
        print(f"    N={N//1024}k: nocache={d.get('nocache')} tcp={d.get('tcp')} "
              f"rdma={d.get('rdma')}  rdma/tcp={rt:.2f}x")
PY
}

case "$LAYER" in
  l1)      run_l1 ;;
  l2)      run_l2 ;;
  l4)      run_l4 ;;
  compare) run_compare ;;
  all)     run_l1; run_l2; run_l4; run_compare ;;
  *) echo "unknown layer '$LAYER' (l1|l2|l4|compare|all)"; exit 2 ;;
esac
echo "[run_benchmark] done: layer=$LAYER transport=$TRANSPORT"
