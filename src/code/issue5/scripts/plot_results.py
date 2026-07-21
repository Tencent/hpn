#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue5 汇总出图 (读 results/raw 的全部 CSV, 出 report/figures/)

三类图 + 模型/QPM 图:
  1) bw_vs_size.png        L1 原语: 有效带宽 vs value 大小 (TCP/RDMA × put/get)
  2) bw_vs_concurrency.png L1 原语: put 带宽 vs 并发 (TCP/RDMA, 展示 RDMA scale)
  3) l2_zerocopy_bw.png    L2 真实 KV: zerocopy vs bytes 的 get 带宽 (TCP/RDMA)
  4) ttft_vs_load.png      L4: P95 TTFT vs 到达率 (三传输 + SLO 线, N=16k 两 regime)
  5) qpm_compare.png       L4: QPM 柱状 (两 regime × 三传输 × 三 N)
  6) protocol_summary.png  一图总览: L1/L2/L4 关键指标的 RDMA/TCP 倍数

用法:
  python3 scripts/plot_results.py --raw results/raw --figdir report/figures
"""
import argparse, csv, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GB = 1 << 30


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def med(rows, pred, col):
    vals = []
    for r in rows:
        if pred(r):
            try:
                vals.append(float(r[col]))
            except (ValueError, KeyError):
                pass
    return statistics.median(vals) if vals else None


# ---- 图1: L1 带宽 vs value 大小 (put/get × tcp/rdma) ----
def plot_bw_vs_size(raw, figdir):
    tcp = read_csv(f"{raw}/transfer_tcp.csv")
    rdma = read_csv(f"{raw}/transfer_rdma.csv")
    if not tcp or not rdma:
        return None
    sizes = ["64MB", "256MB", "1GB"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, op in zip(axes, ("put", "get")):
        for data, name, c in ((tcp, "tcp", "C0"), (rdma, "rdma", "C1")):
            ys = [med(data, lambda r, s=s: r["size"] == s and r["op"] == op
                      and r["success"] == "1", "bandwidth_gbps") for s in sizes]
            ax.plot(sizes, ys, "o-", color=c, label=name)
        ax.set_title(f"L1 {op} bandwidth")
        ax.set_xlabel("value size"); ax.set_ylabel("GB/s")
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle("L1 primitive: effective bandwidth vs value size")
    fig.tight_layout(); fig.savefig(f"{figdir}/bw_vs_size.png", dpi=120); plt.close(fig)
    return "bw_vs_size.png"


# ---- 图2: L1 put 带宽 vs 并发 ----
def plot_bw_vs_conc(raw, figdir):
    tcp = read_csv(f"{raw}/transfer_tcp.csv")
    rdma = read_csv(f"{raw}/transfer_rdma.csv")
    if not tcp or not rdma:
        return None
    concs = ["1", "2", "4", "8"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for data, name, c in ((tcp, "tcp", "C0"), (rdma, "rdma", "C1")):
        ys = [med(data, lambda r, cc=cc: r["concurrency"] == cc and r["op"] == "put"
                  and r["size"] == "64MB" and r["success"] == "1", "bandwidth_gbps")
              for cc in concs]
        ax.plot(concs, ys, "o-", color=c, label=name)
    ax.set_title("L1 put bandwidth vs concurrency (64MB)")
    ax.set_xlabel("concurrency"); ax.set_ylabel("GB/s")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{figdir}/bw_vs_concurrency.png", dpi=120); plt.close(fig)
    return "bw_vs_concurrency.png"


# ---- 图3: L2 真实 KV, zerocopy vs bytes 的 get 带宽 (tcp/rdma) ----
def plot_l2_zerocopy(raw, figdir):
    tcp = read_csv(f"{raw}/kv_tensor_tcp.csv")
    rdma = read_csv(f"{raw}/kv_tensor_rdma.csv")
    if not tcp or not rdma:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, vals, colors = [], [], []
    for data, name in ((tcp, "tcp"), (rdma, "rdma")):
        for path in ("bytes", "zerocopy"):
            v = med(data, lambda r, p=path: r.get("path") == p and r.get("op") == "get"
                    and r.get("success") == "1", "effective_bandwidth_gbps")
            labels.append(f"{name}\n{path}"); vals.append(v or 0)
            colors.append("C0" if name == "tcp" else "C1")
    bars = ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_ylabel("get bandwidth (GB/s)")
    ax.set_title("L2 real-KV get: zerocopy vs bytes (RDMA-zc ~18.7x TCP-zc)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{figdir}/l2_zerocopy_bw.png", dpi=120); plt.close(fig)
    return "l2_zerocopy_bw.png"


# ---- 图4: L4 P95 TTFT vs 到达率 (三传输 + SLO 线), N=16k 两 regime ----
def plot_ttft_vs_load(raw, figdir):
    mod = read_csv(f"{raw}/conc_sweep.csv")
    hi = read_csv(f"{raw}/conc_hihit_sweep.csv")
    if not mod or not hi:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, rows, title in ((axes[0], mod, "moderate hit h=.5"),
                            (axes[1], hi, "high hit h=.95")):
        N = 16384
        slo = None
        for tp, c in (("nocache", "C2"), ("tcp", "C0"), ("rdma", "C1")):
            pts = sorted((float(r["lambda_rps"]), float(r["p95_ttft_ms"]))
                         for r in rows if r["transport"] == tp
                         and int(r["context_length"]) == N)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] / 1e3 for p in pts], "o-",
                    color=c, label=tp, ms=4)
            for r in rows:
                if r["transport"] == tp and int(r["context_length"]) == N:
                    slo = float(r["slo_ms"]) / 1e3
        if slo:
            ax.axhline(slo, ls="--", color="k", alpha=.6, label=f"SLO={slo:.1f}s")
        ax.set_title(f"P95 TTFT vs load (N=16k, {title})")
        ax.set_xlabel("arrival rate λ (req/s)"); ax.set_ylabel("P95 TTFT (s)")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{figdir}/ttft_vs_load.png", dpi=120); plt.close(fig)
    return "ttft_vs_load.png"


# ---- 图5: QPM 柱状 (两 regime × 三传输 × 三 N) ----
def plot_qpm_compare(raw, figdir):
    mod = read_csv(f"{raw}/conc_summary.csv")
    hi = read_csv(f"{raw}/conc_hihit_summary.csv")
    if not mod or not hi:
        return None
    Ns = [8192, 16384, 24576]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, rows, title in ((axes[0], mod, "moderate hit h=.5"),
                            (axes[1], hi, "high hit h=.95")):
        qpm = {}
        for r in rows:
            qpm[(int(r["context_length"]), r["transport"])] = float(r["qpm"])
        x = np.arange(len(Ns)); w = 0.25
        for i, (tp, c) in enumerate((("nocache", "C2"), ("tcp", "C0"), ("rdma", "C1"))):
            ys = [qpm.get((N, tp), 0) for N in Ns]
            ax.bar(x + (i - 1) * w, ys, w, label=tp, color=c)
        ax.set_xticks(x); ax.set_xticklabels([f"{N // 1024}k" for N in Ns])
        ax.set_title(f"QPM ({title})"); ax.set_xlabel("context N"); ax.set_ylabel("QPM")
        ax.grid(alpha=.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{figdir}/qpm_compare.png", dpi=120); plt.close(fig)
    return "qpm_compare.png"


# ---- 图6: 一图总览 RDMA/TCP 倍数 (L1 put / L2 zc get / L4 QPM 两 regime) ----
def plot_protocol_summary(raw, figdir):
    t1 = read_csv(f"{raw}/transfer_tcp.csv"); r1 = read_csv(f"{raw}/transfer_rdma.csv")
    t2 = read_csv(f"{raw}/kv_tensor_tcp.csv"); r2 = read_csv(f"{raw}/kv_tensor_rdma.csv")
    mod = read_csv(f"{raw}/conc_summary.csv"); hi = read_csv(f"{raw}/conc_hihit_summary.csv")
    ratios, labels = [], []

    def peak(rows, op):
        vals = [float(x["bandwidth_gbps"]) for x in rows
                if x["op"] == op and x["success"] == "1"]
        return max(vals) if vals else None
    if t1 and r1:
        ratios.append(peak(r1, "put") / peak(t1, "put")); labels.append("L1 put\n(peak)")
    if t2 and r2:
        zt = med(t2, lambda r: r.get("path") == "zerocopy" and r.get("op") == "get"
                 and r.get("success") == "1", "effective_bandwidth_gbps")
        zr = med(r2, lambda r: r.get("path") == "zerocopy" and r.get("op") == "get"
                 and r.get("success") == "1", "effective_bandwidth_gbps")
        if zt and zr:
            ratios.append(zr / zt); labels.append("L2 zc get\n(median)")

    def qpm_ratio(rows, N):
        d = {r["transport"]: float(r["qpm"]) for r in rows
             if int(r["context_length"]) == N}
        return d["rdma"] / d["tcp"] if d.get("tcp") else None
    if mod:
        ratios.append(qpm_ratio(mod, 16384)); labels.append("L4 QPM\nmod-hit 16k")
    if hi:
        ratios.append(qpm_ratio(hi, 16384)); labels.append("L4 QPM\nhi-hit 16k")
    if not ratios:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(ratios)), ratios, color=["C1", "C1", "C3", "C3"][:len(ratios)])
    ax.axhline(1.0, ls="--", color="k", alpha=.5)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RDMA / TCP (×)"); ax.set_yscale("log")
    ax.set_title("RDMA vs TCP: transport wins big, e2e QPM depends on retrieve share")
    for b, v in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=.3, axis="y", which="both")
    fig.tight_layout(); fig.savefig(f"{figdir}/protocol_summary.png", dpi=120); plt.close(fig)
    return "protocol_summary.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/raw")
    ap.add_argument("--figdir", default="report/figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    made = []
    for fn in (plot_bw_vs_size, plot_bw_vs_conc, plot_l2_zerocopy,
               plot_ttft_vs_load, plot_qpm_compare, plot_protocol_summary):
        try:
            r = fn(args.raw, args.figdir)
            if r:
                made.append(r)
        except Exception as e:
            print(f"[skip] {fn.__name__}: {e}")
    print(f"[out] wrote {made} -> {args.figdir}")


if __name__ == "__main__":
    main()
