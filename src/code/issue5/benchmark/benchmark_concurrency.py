#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
背景: 端到端命中路径被 write_through kernel 卡死, 无法在真 server 上跑并发
命中。用实测数值驱动的离散事件重放(replay): 每个逻辑请求的服务时间由
  - 命中: T_lookup + T_restore(hN)/B_retrieve + T_prefill((1-h)N)   [远端 retrieve + 后缀现算]
  - 未命中: T_prefill(N)                                              [全量重算]
其中 T_prefill(N) 用步骤5 从 L3 prefill_local.csv 拟合的 a·N+b·N² (ms),
T_restore 用 L2 kv_tensor 的 zerocopy get 带宽(TCP~1.1 / RDMA~20.7 GB/s)。

排队模型: c 个 prefill worker 的 FIFO 队列(GPU 算力受限, 默认 c=1 即算力串行),
Poisson 到达(率 λ req/s)。扫 λ 直到 P95 TTFT 超过 SLO -> λ_max; QPM=60·λ_max;
成本/query ∝ 固定节点机时 / QPM (归一到 no-cache 基线)。

SLO 对 TCP/RDMA/nocache **三者相同**(取 no-cache 单请求全量重算服务时间 × slo-factor),
保证公平对比。对每 context 各扫一遍。

用法 :
  python3 benchmark/benchmark_concurrency.py \
    --calib results/raw/prefill_local.csv \
    --transfer-tcp results/raw/kv_tensor_tcp.csv \
    --transfer-rdma results/raw/kv_tensor_rdma.csv \
    --context 16k,24k --hit-prefix 0.5 --phit 0.5 --slo-factor 1.5 \
    --out-prefix results/raw/conc
"""
import argparse, csv, math, os, random, statistics, sys

GB = 1 << 30
S_TOKEN_V32 = 61 * (512 + 64) * 2   # 70272 B = 68.6 KiB/token (V3.2 MLA 单 latent)
PAGE = 64


def align(n, page=PAGE):
    return max(0, (n // page) * page)


def parse_ctx(s):
    out = []
    for t in s.replace(" ", "").lower().split(","):
        if not t:
            continue
        out.append(int(float(t[:-1]) * 1024) if t.endswith("k") else int(t))
    return out


def fit_tprefill(path):
    """从 L3 CSV 最小二乘拟合 T_prefill(N)=a*N+b*N^2 (过原点, ms)。"""
    import numpy as np
    byN = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            byN.setdefault(int(r["context_length"]), []).append(float(r["recompute_ms"]))
    pts = sorted((N, statistics.median(v)) for N, v in byN.items())
    Ns = np.array([p[0] for p in pts], float)
    Ts = np.array([p[1] for p in pts], float)
    A = np.vstack([Ns, Ns ** 2]).T
    coef, *_ = np.linalg.lstsq(A, Ts, rcond=None)
    return float(coef[0]), float(coef[1]), pts


def load_retrieve_bw(path, want="zerocopy", op="get"):
    bws = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("path") == want and r.get("op") == op and r.get("success") == "1":
                try:
                    bws.append(float(r["effective_bandwidth_gbps"]))
                except (ValueError, KeyError):
                    pass
    return statistics.median(bws) if bws else None


def t_prefill(N, a, b):
    return a * N + b * N * N


def service_ms(N, is_hit, h, a, b, b_ret_gbps, lookup_ms):
    """单请求服务时间(ms)。命中: lookup+restore+后缀现算; 未命中: 全量重算。"""
    if not is_hit:
        return t_prefill(N, a, b)
    hN = align(int(round(h * N)))
    uniq = N - hN
    s_kv_gb = hN * S_TOKEN_V32 / GB
    t_restore = (s_kv_gb / b_ret_gbps) * 1e3 if b_ret_gbps else 0.0
    return lookup_ms + t_restore + t_prefill(uniq, a, b)


def simulate(lam_rps, n_req, workers, svc_fn, rng, warmup_frac=0.1):
    """c-worker FIFO 队列 + Poisson 到达的离散事件重放, 返回 TTFT 列表(ms, 去 warmup)。
    FIFO+c 同构 server: 每到达分配给最早空闲的 worker。"""
    free = [0.0] * workers            # 各 worker 空闲时刻(ms)
    t = 0.0
    ttfts = []
    for i in range(n_req):
        t += rng.expovariate(lam_rps) * 1e3          # 到达间隔(ms)
        w = min(range(workers), key=lambda k: free[k])
        start = max(t, free[w])
        svc = svc_fn()
        finish = start + svc
        free[w] = finish
        ttfts.append(finish - t)                     # TTFT = 排队等待 + 服务
    cut = int(n_req * warmup_frac)
    return ttfts[cut:]


def pctl(xs, p):
    import numpy as np
    return float(np.percentile(xs, p)) if xs else float("inf")


def make_svc_fn(N, phit, h, a, b, b_ret, lookup_ms, rng):
    """返回一个无参函数: 每次按 phit 概率抽命中/未命中, 给出服务时间(ms)。"""
    def fn():
        is_hit = (rng.random() < phit)
        return service_ms(N, is_hit, h, a, b, b_ret, lookup_ms)
    return fn


def sweep_lambda(N, phit, h, a, b, b_ret, lookup_ms, slo_ms, n_req, workers,
                 lam_grid, seed):
    """扫 λ, 返回 (λ_max_rps, per_lambda_rows)。λ_max = P95 TTFT ≤ SLO 的最大 λ。"""
    rows, lam_max = [], 0.0
    for lam in lam_grid:
        rng = random.Random(seed)              # 同 seed 保证各 λ 间可比
        svc = make_svc_fn(N, phit, h, a, b, b_ret, lookup_ms, rng)
        ttfts = simulate(lam, n_req, workers, svc, rng)
        p50, p95, p99 = pctl(ttfts, 50), pctl(ttfts, 95), pctl(ttfts, 99)
        ok = p95 <= slo_ms
        if ok:
            lam_max = lam
        rows.append(dict(lambda_rps=round(lam, 4), p50_ttft_ms=round(p50, 1),
                         p95_ttft_ms=round(p95, 1), p99_ttft_ms=round(p99, 1),
                         slo_ms=round(slo_ms, 1), within_slo=int(ok)))
        if not ok and lam_max > 0:
            break                              # 已越过 SLO, 后续更高 λ 无意义
    return lam_max, rows


def build_grid(base_svc_ms, workers):
    """λ 网格: 以单 worker 饱和率 (workers*1000/base_svc) 为量程, 细扫到 ~1.3×。"""
    sat = workers * 1000.0 / base_svc_ms       # 名义饱和到达率(req/s)
    return [sat * f for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8,
                              0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/raw/prefill_local.csv")
    ap.add_argument("--transfer-tcp", default="results/raw/kv_tensor_tcp.csv")
    ap.add_argument("--transfer-rdma", default="results/raw/kv_tensor_rdma.csv")
    ap.add_argument("--context", default="8k,16k,24k")
    ap.add_argument("--hit-prefix", type=float, default=0.5, help="命中请求复用的前缀比例 h")
    ap.add_argument("--phit", type=float, default=0.5, help="请求命中缓存的概率")
    ap.add_argument("--workers", type=int, default=1, help="并行 prefill worker 数 (算力串行=1)")
    ap.add_argument("--lookup-ms", type=float, default=5.0, help="远端查表固定开销")
    ap.add_argument("--slo-factor", type=float, default=1.5, help="SLO = nocache 单请求服务 × 此")
    ap.add_argument("--n-req", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-prefix", default="results/raw/conc")
    args = ap.parse_args()

    a, b, pts = fit_tprefill(args.calib)
    bw = {"nocache": None,
          "tcp": load_retrieve_bw(args.transfer_tcp),
          "rdma": load_retrieve_bw(args.transfer_rdma)}
    print(f"[fit] T_prefill(N)={a:.5f}·N+{b:.3e}·N²  |  "
          f"B_retrieve tcp={bw['tcp']:.2f} rdma={bw['rdma']:.2f} GB/s", flush=True)
    print(f"[cfg] hit_prefix h={args.hit_prefix} phit={args.phit} workers={args.workers} "
          f"lookup={args.lookup_ms}ms slo_factor={args.slo_factor}", flush=True)

    ctxs = parse_ctx(args.context)
    all_rows, summary = [], []
    for N in ctxs:
        # SLO 由 no-cache 全量重算服务时间定, 三种传输共用同一 SLO (公平)
        base = t_prefill(N, a, b)
        slo = base * args.slo_factor
        # 平均服务时间 (用于 λ 网格量程): 取 nocache 与 hit 的期望
        for transport, b_ret in bw.items():
            # nocache = 缓存关闭 => 每请求都全量重算 (phit_eff=0); 有缓存 => 用真实 phit
            phit_eff = 0.0 if transport == "nocache" else args.phit
            mean_svc = (phit_eff * service_ms(N, True, args.hit_prefix, a, b,
                                              b_ret, args.lookup_ms)
                        + (1 - phit_eff) * t_prefill(N, a, b))
            grid = build_grid(mean_svc, args.workers)
            lam_max, rows = sweep_lambda(N, phit_eff, args.hit_prefix, a, b, b_ret,
                                         args.lookup_ms, slo, args.n_req, args.workers,
                                         grid, args.seed)
            qpm = 60.0 * lam_max
            for r in rows:
                r.update(transport=transport, context_length=N, hit_prefix=args.hit_prefix,
                         phit=phit_eff, workers=args.workers)
                all_rows.append(r)
            summary.append(dict(context_length=N, transport=transport,
                                mean_service_ms=round(mean_svc, 1), slo_ms=round(slo, 1),
                                lambda_max_rps=round(lam_max, 4), qpm=round(qpm, 2)))
            print(f"  N={N//1024}k {transport:8} mean_svc={mean_svc:7.0f}ms "
                  f"SLO={slo:7.0f}ms  λ_max={lam_max:.4f}rps  QPM={qpm:.2f}", flush=True)

    # 每 context 打印 TCP/RDMA 相对 no-cache 的 QPM 增益 + RDMA vs TCP 成本比
    print("\n[对比] QPM 增益 (相对 no-cache) 与 成本/query (∝ 1/QPM, 归一 nocache):")
    byN = {}
    for s in summary:
        byN.setdefault(s["context_length"], {})[s["transport"]] = s
    for N, d in sorted(byN.items()):
        base_qpm = d["nocache"]["qpm"] or 1e-9
        line = f"  N={N//1024}k:"
        for tp in ("nocache", "tcp", "rdma"):
            q = d[tp]["qpm"]
            gain = q / base_qpm
            cost = base_qpm / q if q > 0 else float("inf")
            line += f"  {tp}=QPM {q:.1f}(×{gain:.2f}, cost×{cost:.2f})"
        # RDMA vs TCP
        if d["tcp"]["qpm"] > 0:
            line += f"  || rdma/tcp QPM={d['rdma']['qpm']/d['tcp']['qpm']:.2f}×"
        print(line, flush=True)

    _write(args.out_prefix + "_sweep.csv", all_rows,
           ["transport", "context_length", "hit_prefix", "phit", "workers",
            "lambda_rps", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "slo_ms", "within_slo"])
    _write(args.out_prefix + "_summary.csv", summary,
           ["context_length", "transport", "mean_service_ms", "slo_ms", "lambda_max_rps", "qpm"])
    print(f"\n[out] {args.out_prefix}_sweep.csv ({len(all_rows)} rows) + "
          f"{args.out_prefix}_summary.csv ({len(summary)} rows)", flush=True)


def _write(path, rows, cols):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()


