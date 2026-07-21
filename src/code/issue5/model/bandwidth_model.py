#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue5 带宽临界值模型 B_crit + 用 L1/L2/L3 实测标定

闭式解 (PLAN 附录 A):
  T_full(N)      = T_prefill(N)
  T_cache(N,h,B) = T_0 + S_KV(hN)/B + T_restore(hN) + T_prefill((1-h)N)
  令 T_cache < T_full 解出临界带宽:
  B_crit(N,h)    = S_KV(hN) / ( T_prefill(N) - T_prefill((1-h)N) - T_lookup - T_restore - L_net )
  划算条件: 实际 retrieve 带宽 B_retrieve > B_crit  <=>  以存代算比全量重算快。

本脚本做四件事:
 1) 用 L3 prefill_local.csv 拟合算力侧 T_prefill(N)=a*N+b*N^2 (prefill 前向墙钟 ms, 过原点)。
    b 项吸收 attention 的超线性; NSA 稀疏(topk=2048)会压平, 故 128k 外推给区间上界(见 doc)。
 2) 每 (N,h) 用实测 recompute/cold_uniq 直接算 B_crit(本地命中 restore/lookup≈0),
    再用拟合模型算一遍 -> 标定误差 (model vs measured)。
 3) 用 L1/L2 的 B_retrieve(TCP/RDMA zerocopy) 算 T_restore、净收益、是否划算。
 4) 外推线上 128K/30% 命中: 预测 B_crit, 对比线上 TCP~2.5GB/s / 实测 TCP / RDMA。
 出 results/model_fit.csv + 拟合图 results/figures/。

用法:
  python3 model/bandwidth_model.py \
    --calib results/raw/prefill_local.csv \
    --transfer-tcp results/raw/kv_tensor_tcp.csv \
    --transfer-rdma results/raw/kv_tensor_rdma.csv \
    --report results/model_fit.csv --figdir results/figures
"""
import argparse, csv, os, sys, statistics

GB = 1 << 30
KiB = 1 << 10
# V3.2 MLA 单 latent: 每 token KV = L*(kv_lora_rank+qk_rope)*b (与 L2/L3 脚本一致)
S_TOKEN_V32 = 61 * (512 + 64) * 2   # = 70272 B = 68.6 KiB/token
PAGE = 64


def s_kv_bytes(n_tok):
    return n_tok * S_TOKEN_V32


def align(n, page=PAGE):
    return max(0, (n // page) * page)


def load_prefill(path):
    """读 L3 CSV -> list[dict(N,h,cached,recompute_ms,cold_uniq_ms,total_ms,saved_ms)]。"""
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                N=int(r["context_length"]), h=float(r["hit_ratio"]),
                cached=int(r["cached_tokens"]),
                recompute_ms=float(r["recompute_ms"]),
                cold_uniq_ms=float(r["cold_uniq_ms"]),
                total_ms=float(r["total_ms"]),
                saved_ms=float(r["saved_ms"])))
    return rows


def fit_tprefill(rows):
    """最小二乘拟合 T_prefill(N)=a*N+b*N^2 (过原点)。每个 N 用 h=0 的 recompute 均值做标定点。
    返回 (a[ms/tok], b[ms/tok^2], r2, points)。"""
    import numpy as np
    # 每个 N 的全量重算基线 = 该 N 所有行的 recompute_ms 中位数 (h 无关, 都是全量现算)
    byN = {}
    for r in rows:
        byN.setdefault(r["N"], []).append(r["recompute_ms"])
    pts = sorted((N, statistics.median(v)) for N, v in byN.items())
    Ns = np.array([p[0] for p in pts], dtype=float)
    Ts = np.array([p[1] for p in pts], dtype=float)
    # 设计矩阵 [N, N^2], 无截距 (N=0 -> 0 prefill)
    A = np.vstack([Ns, Ns ** 2]).T
    coef, *_ = np.linalg.lstsq(A, Ts, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((Ts - pred) ** 2).sum())
    ss_tot = float(((Ts - Ts.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2, pts


def t_prefill(N, a, b):
    return a * N + b * N * N


def load_retrieve_bw(path, want_path="zerocopy", op="get"):
    """从 L2 kv_tensor CSV 取 retrieve 有效带宽 (GB/s) 的中位数。"""
    bws = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("path") == want_path and r.get("op") == op and r.get("success") == "1":
                try:
                    bws.append(float(r["effective_bandwidth_gbps"]))
                except (ValueError, KeyError):
                    pass
    return statistics.median(bws) if bws else None


def b_crit_gbps(s_kv_B, denom_ms):
    """B_crit = S_KV / 分母(转秒). 分母<=0 => 不可能划算 (返回 inf)。"""
    if denom_ms <= 0:
        return float("inf")
    return (s_kv_B / GB) / (denom_ms / 1e3)


def compute_cells(rows, a, b, retrieves):
    """对每 (N,h) 生成: 实测 B_crit + 模型 B_crit + 每种传输的 T_restore/净收益/是否划算。"""
    out = []
    for r in rows:
        N, h, cached = r["N"], r["h"], r["cached"]
        if cached <= 0:
            continue  # h=0 无前缀可复用, B_crit 无意义
        s_kv = s_kv_bytes(cached)
        # 实测分母 = 前缀现算时间 = 全量重算 - 冷唯一后缀 (本地命中 lookup/restore≈0)
        denom_meas = r["recompute_ms"] - r["cold_uniq_ms"]
        bcrit_meas = b_crit_gbps(s_kv, denom_meas)
        # 模型分母 = T_prefill(N) - T_prefill((1-h)N)
        uniq = N - align(int(round(h * N)))
        denom_model = t_prefill(N, a, b) - t_prefill(uniq, a, b)
        bcrit_model = b_crit_gbps(s_kv, denom_model)
        base = dict(N=N, h=h, cached=cached, s_kv_gb=round(s_kv / GB, 4),
                    tprefill_N_ms=round(t_prefill(N, a, b), 1),
                    denom_meas_ms=round(denom_meas, 1),
                    denom_model_ms=round(denom_model, 1),
                    bcrit_meas_gbps=round(bcrit_meas, 4),
                    bcrit_model_gbps=round(bcrit_model, 4),
                    measured_saved_ms=r["saved_ms"])
        for tname, bw in retrieves.items():
            t_restore = (s_kv / GB) / bw * 1e3      # ms
            net = denom_meas - t_restore            # 净省下的时间 (算侧收益 - restore 成本)
            row = dict(base, transport=tname, b_retrieve_gbps=round(bw, 3),
                       t_restore_ms=round(t_restore, 1),
                       net_saved_ms=round(net, 1),
                       profitable=int(bw > bcrit_meas))
            out.append(row)
    return out


def extrapolate(a, b, retrieves, N=131072, h=0.30, online_tcp=2.5):
    """外推线上场景 (默认 128K / 30% 命中)。返回 dict。"""
    hN = align(int(round(h * N)))
    uniq = N - hN
    s_kv = s_kv_bytes(hN)
    denom = t_prefill(N, a, b) - t_prefill(uniq, a, b)
    bcrit = b_crit_gbps(s_kv, denom)
    res = dict(N=N, h=h, hit_tokens=hN, s_kv_gb=round(s_kv / GB, 3),
               tprefill_N_ms=round(t_prefill(N, a, b), 0),
               tprefill_uniq_ms=round(t_prefill(uniq, a, b), 0),
               denom_ms=round(denom, 0), bcrit_gbps=round(bcrit, 4))
    bws = dict(retrieves); bws["online_tcp_hypo"] = online_tcp
    for tname, bw in bws.items():
        res[f"Trestore_{tname}_ms"] = round((s_kv / GB) / bw * 1e3, 0)
        res[f"margin_{tname}_x"] = round(bw / bcrit, 1) if bcrit > 0 else float("inf")
    return res


def write_csv(path, cells):
    cols = ["N", "h", "cached", "s_kv_gb", "tprefill_N_ms", "denom_meas_ms",
            "denom_model_ms", "bcrit_meas_gbps", "bcrit_model_gbps", "transport",
            "b_retrieve_gbps", "t_restore_ms", "net_saved_ms", "profitable",
            "measured_saved_ms"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for c in cells:
            w.writerow(c)


def make_plots(figdir, rows, cells, a, b, retrieves, pts):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(figdir, exist_ok=True)

    # 图1: T_prefill 拟合 vs 实测
    fig, ax = plt.subplots(figsize=(6, 4))
    Ns = np.array([p[0] for p in pts]); Ts = np.array([p[1] for p in pts])
    xx = np.linspace(0, 131072, 200)
    ax.scatter(Ns / 1024, Ts / 1e3, c="C3", zorder=3, label="measured recompute")
    ax.plot(xx / 1024, t_prefill(xx, a, b) / 1e3, "C0-",
            label=f"fit a·N+b·N²\na={a:.4f}ms/tok\nb={b:.2e}ms/tok²")
    ax.set_xlabel("context N (K tokens)"); ax.set_ylabel("T_prefill (s)")
    ax.set_title("V3.2 prefill compute cost (fit)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{figdir}/tprefill_fit.png", dpi=120); plt.close(fig)

    # 图2: B_crit vs h (各 N), 叠加传输带宽水平线
    fig, ax = plt.subplots(figsize=(6, 4))
    byN = {}
    for c in cells:
        if c["transport"] == list(retrieves)[0]:
            byN.setdefault(c["N"], []).append((c["h"], c["bcrit_meas_gbps"]))
    for N, hv in sorted(byN.items()):
        hv.sort(); ax.plot([x[0] for x in hv], [x[1] for x in hv],
                           "o-", label=f"N={N//1024}k")
    for tname, bw in retrieves.items():
        ax.axhline(bw, ls="--", alpha=.6, label=f"{tname} B_ret={bw:.1f}")
    ax.set_yscale("log"); ax.set_xlabel("hit ratio h"); ax.set_ylabel("B_crit (GB/s, log)")
    ax.set_title("B_crit vs hit ratio (all << transport BW)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{figdir}/bcrit_vs_hit.png", dpi=120); plt.close(fig)

    # 图3: T_restore TCP vs RDMA (各 N,h=1.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    hit1 = [c for c in cells if c["h"] == 1.0]
    Nl = sorted({c["N"] for c in hit1})
    xs = range(len(Nl)); w = 0.35
    for i, tname in enumerate(retrieves):
        ys = [next(c["t_restore_ms"] for c in hit1 if c["N"] == N and c["transport"] == tname)
              for N in Nl]
        ax.bar([x + (i - .5) * w for x in xs], ys, w, label=tname)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{N//1024}k" for N in Nl])
    ax.set_xlabel("context N"); ax.set_ylabel("T_restore (ms), h=1.0")
    ax.set_title("Restore latency: TCP vs RDMA"); ax.legend(); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{figdir}/trestore_tcp_vs_rdma.png", dpi=120); plt.close(fig)
    return ["tprefill_fit.png", "bcrit_vs_hit.png", "trestore_tcp_vs_rdma.png"]


def selftest(a, b, retrieves):
    """闭式解一致性自检: T_cache < T_full  <=>  B_retrieve > B_crit。"""
    N, h = 16384, 0.5
    hN = align(int(round(h * N))); uniq = N - hN
    s_kv = s_kv_bytes(hN)
    denom = t_prefill(N, a, b) - t_prefill(uniq, a, b)
    bcrit = b_crit_gbps(s_kv, denom)
    for bw in (bcrit * 0.5, bcrit * 2.0):   # 一半 / 两倍临界带宽
        t_restore = (s_kv / GB) / bw * 1e3
        t_cache = t_prefill(uniq, a, b) + t_restore     # T_0=lookup=Lnet=0
        t_full = t_prefill(N, a, b)
        profitable = t_cache < t_full
        assert profitable == (bw > bcrit), \
            f"selftest FAIL bw={bw:.3f} bcrit={bcrit:.3f} cache={t_cache:.0f} full={t_full:.0f}"
    print("[selftest] closed-form B_crit <-> T_cache<T_full consistent OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/raw/prefill_local.csv")
    ap.add_argument("--transfer-tcp", default="results/raw/kv_tensor_tcp.csv")
    ap.add_argument("--transfer-rdma", default="results/raw/kv_tensor_rdma.csv")
    ap.add_argument("--report", default="results/model_fit.csv")
    ap.add_argument("--figdir", default="results/figures")
    ap.add_argument("--online-tcp", type=float, default=2.5, help="线上 TCP 带宽假设 (GB/s)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    rows = load_prefill(args.calib)
    a, b, r2, pts = fit_tprefill(rows)
    print(f"[fit] T_prefill(N) = {a:.5f}·N + {b:.3e}·N²  ms   (R²={r2:.4f})", flush=True)
    print(f"[fit] 标定点: " + ", ".join(f"{N//1024}k={T:.0f}ms" for N, T in pts), flush=True)
    print(f"[fit] 每 token 算力代价 8k→24k: "
          + ", ".join(f"{N//1024}k={T/N:.3f}ms/tok" for N, T in pts), flush=True)

    retrieves = {}
    tcp = load_retrieve_bw(args.transfer_tcp)
    rdma = load_retrieve_bw(args.transfer_rdma)
    if tcp:  retrieves["tcp_zc"] = tcp
    if rdma: retrieves["rdma_zc"] = rdma
    print(f"[bw] retrieve (L2 zerocopy get 中位): "
          + ", ".join(f"{k}={v:.2f}GB/s" for k, v in retrieves.items()), flush=True)

    cells = compute_cells(rows, a, b, retrieves)
    write_csv(args.report, cells)
    print(f"[out] wrote {len(cells)} rows -> {args.report}", flush=True)

    # 标定误差: 模型 B_crit vs 实测 B_crit
    errs = [abs(c["bcrit_model_gbps"] - c["bcrit_meas_gbps"]) / c["bcrit_meas_gbps"]
            for c in cells if c["transport"] == list(retrieves)[0] and c["bcrit_meas_gbps"] > 0]
    if errs:
        print(f"[calib] 模型 vs 实测 B_crit 相对误差: "
              f"median={statistics.median(errs)*100:.1f}% max={max(errs)*100:.1f}%", flush=True)

    print("\n[B_crit 实测范围] "
          f"{min(c['bcrit_meas_gbps'] for c in cells):.3f} – "
          f"{max(c['bcrit_meas_gbps'] for c in cells):.3f} GB/s", flush=True)

    ex = extrapolate(a, b, retrieves, N=131072, h=0.30, online_tcp=args.online_tcp)
    print("\n[外推 128K / 30% 命中] (验收① 验证点)")
    print(f"  hit_tokens={ex['hit_tokens']}  S_KV={ex['s_kv_gb']}GB  "
          f"T_prefill(128k)≈{ex['tprefill_N_ms']/1e3:.1f}s  分母≈{ex['denom_ms']/1e3:.1f}s")
    print(f"  => 预测 B_crit = {ex['bcrit_gbps']:.4f} GB/s")
    for tname in list(retrieves) + ["online_tcp_hypo"]:
        print(f"     {tname:16} margin={ex[f'margin_{tname}_x']}×  "
              f"T_restore={ex[f'Trestore_{tname}_ms']/1e3:.2f}s")

    selftest(a, b, retrieves)

    if not args.no_plot:
        try:
            figs = make_plots(args.figdir, rows, cells, a, b, retrieves, pts)
            print(f"[fig] wrote {figs} -> {args.figdir}", flush=True)
        except Exception as e:
            print(f"[fig] skip plot: {e}", flush=True)


if __name__ == "__main__":
    main()
