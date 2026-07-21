#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue5 prefill_only end to end microbench

prefill_only(max_new_tokens=0)在本地 device radix cache 下:
  - 全量重算 T_prefill(N)  = 冷 prefill 长度 N。  -> recompute 基线, 喂 B_crit 分母。
  - 前缀本地命中 total(N,h) = 先发 shared_prefix(hN) 预热进 radix, 不 flush, 再发
                              shared_prefix + unique_suffix((1-h)N); cached_tokens=hN,
                              prefix 的 KV 直接复用(device 本地, restore≈0), 只现算后缀。
  - saved(N,h) = T_prefill(N) - total(N,h)  = 复用 hN 前缀 KV 省下的**计算**时间。
这是「以存代算」的**算力节省侧**真实标定(传输侧由 L2 给); 本地命中 => transfer/restore≈0。

命中率 h: shared_prefix 长度 = align(round(h*N), page); unique_suffix = N - prefix。
JIT/无 cuda graph => 每个 shape 首次请求有编译毛刺, 故每 shape 先 warmup 丢弃, 再取 median。

用法(prefill pod 内, server 已 ready, 用 start_prefill_v32_nohicache.sh 起):
  export PATH=/root/miniconda/envs/python310_torch29_cuda/bin:$PATH
  python3 benchmark_prefill.py --base http://$(hostname -i):8000 --mode local \
     --transport local_radix --context 8k,32k,128k --hit 0,0.25,0.5,0.75,1.0 \
     --iters 3 --warmup 1 --out /root/prefill_local.csv
"""
import argparse, os, json, time, statistics, csv, sys, random, urllib.request

GB = 1 << 30
KiB = 1 << 10
# DeepSeek-V3.2 MLA: 单 latent, 每 token KV = L*(kv_lora_rank+qk_rope)*b
V32 = dict(num_layers=61, kv_lora_rank=512, qk_rope=64, dtype_bytes=2)


def s_token_v32():
    return V32["num_layers"] * (V32["kv_lora_rank"] + V32["qk_rope"]) * V32["dtype_bytes"]


def parse_ctx(s):
    out = []
    for tok in s.replace(" ", "").lower().split(","):
        if not tok:
            continue
        out.append(int(float(tok[:-1]) * 1024) if tok.endswith("k") else int(tok))
    return out


def post(base, path, obj, timeout=1800):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    dt = time.perf_counter() - t
    try:
        return dt, json.loads(body)
    except Exception:
        return dt, {"_raw": body[:300].decode(errors="replace")}


def flush(base):
    # /flush_cache 清本地 radix(host+device)。注意: 有在跑的请求时返回 400, 故只在 idle 调。
    try:
        urllib.request.urlopen(urllib.request.Request(base + "/flush_cache", method="POST"),
                               timeout=60).read()
    except Exception:
        pass
    time.sleep(0.4)


def cached_tokens(meta):
    m = meta.get("meta_info", meta) if isinstance(meta, dict) else {}
    return int(m.get("cached_tokens", -1)), int(m.get("prompt_tokens", -1))


def align(n, page):
    return max(0, (n // page) * page)


def make_ids(n, seed, vocab_lo=10, vocab_hi=100000):
    rng = random.Random(seed)
    return [rng.randint(vocab_lo, vocab_hi) for _ in range(n)]


def gen_prefill(base, ids, timeout=1800):
    """发一个 prefill_only 请求 (max_new_tokens=0), 返回 (e2e_s, cached, prompt)。"""
    body = {"input_ids": ids, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
            "stream": False, "return_logprob": False}
    dt, resp = post(base, "/generate", body, timeout=timeout)
    c, p = cached_tokens(resp)
    return dt, c, p, resp


def median_fresh(base, n, page, iters, seed0):
    """冷全量 prefill 长度 n 的 median 墙钟(ms)。每次 flush + 唯一内容 => 真现算; shape 已预热。"""
    ts = []
    for it in range(iters):
        flush(base)
        dt, c, p, _ = gen_prefill(base, make_ids(n, seed0 + it))
        ts.append(dt * 1e3)
    return statistics.median(ts)


def warm_shape(base, n, seed):
    """对某个长度 n 先发一个 throwaway 请求, 触发 JIT 编译(--disable-cuda-graph 首次有毛刺)。"""
    flush(base)
    try:
        gen_prefill(base, make_ids(n, seed))
    except Exception:
        pass


def bench_one(base, N, h, page, iters, warmup, st, rows, transport, mode):
    Lhit = align(N, page) if h >= 1.0 else align(int(round(h * N)), page)
    Luniq = N - Lhit
    prefix = make_ids(Lhit, seed=42) if Lhit > 0 else []

    # 预热 shape N (JIT), 然后测 recompute 基线 = 冷全量 T_prefill(N)
    if warmup:
        warm_shape(base, N, seed=90000)
    cold_full_ms = median_fresh(base, N, page, iters, seed0=90001)

    # 冷 unique 部分 (1-h)N: 命中时仍需现算的后缀, 作为下界参考
    if Luniq > 0:
        if warmup:
            warm_shape(base, Luniq, seed=91000)
        cold_uniq_ms = median_fresh(base, Luniq, page, iters, seed0=91001)
    else:
        cold_uniq_ms = 0.0

    hit_ms_list, cached_list, prompt_list = [], [], []
    if Lhit > 0:
        # 预热前缀进本地 radix (不 flush), 再预热一次 prefix+suffix 的完整 shape
        flush(base)
        gen_prefill(base, prefix)                       # warm prefix KV into radix (untimed)
        if warmup:
            gen_prefill(base, prefix + (make_ids(Luniq, seed=69999) if Luniq > 0 else []))
        for it in range(iters):
            # 每 iter 唯一后缀; **不 flush** => prefix 一直在本地 radix, cached=Lhit
            suffix = make_ids(Luniq, seed=70000 + it) if Luniq > 0 else []
            dt, c, p, _ = gen_prefill(base, prefix + suffix)
            hit_ms_list.append(dt * 1e3)
            cached_list.append(c)
            prompt_list.append(p)
    else:
        # h=0: 命中即 cold_full 自身 (无前缀可复用)
        if warmup:
            warm_shape(base, N, seed=79999)
        for it in range(iters):
            flush(base)
            dt, c, p, _ = gen_prefill(base, make_ids(N, seed=80000 + it))
            hit_ms_list.append(dt * 1e3)
            cached_list.append(c)
            prompt_list.append(p)

    hit_ms = statistics.median(hit_ms_list)
    cached = int(statistics.median(cached_list)) if cached_list else -1
    prompt = int(statistics.median(prompt_list)) if prompt_list else -1
    kv_bytes = st * max(cached, 0)
    # 本地命中: prefix KV 已在 device, restore≈0; saved = 全量重算 - 命中总时 = 省下的算力时间
    restore_ms = 0.0
    saved_ms = max(cold_full_ms - hit_ms, 0.0)
    # 本地命中传输为 0 => 有效带宽不适用(留空); 远端 retrieve 带宽由 L2 给
    eff_bw = ""

    rows.append(dict(
        transport=transport, context_length=N, hit_ratio=h, concurrency=1,
        prefix_tokens=Lhit, uniq_tokens=Luniq, cached_tokens=cached, prompt_tokens=prompt,
        kv_bytes=kv_bytes, s_token=st,
        lookup_ms="", transfer_ms=0.0, restore_ms=restore_ms,
        cold_uniq_ms=round(cold_uniq_ms, 2), recompute_ms=round(cold_full_ms, 2),
        saved_ms=round(saved_ms, 2), total_ms=round(hit_ms, 2),
        effective_bandwidth_gbps=eff_bw, max_abs_error="", success=1))
    print(f"  {transport:10} N={N:<7} h={h:<4} prefix={Lhit:<7} cached={cached:<7} "
          f"total={hit_ms:.0f}ms recompute={cold_full_ms:.0f}ms saved={saved_ms:.0f}ms "
          f"cold_uniq={cold_uniq_ms:.0f}ms", flush=True)


def write_csv(path, rows):
    if not rows:
        return
    cols = ["transport", "context_length", "hit_ratio", "concurrency", "prefix_tokens",
            "uniq_tokens", "cached_tokens", "prompt_tokens", "kv_bytes", "s_token",
            "lookup_ms", "transfer_ms", "restore_ms", "cold_uniq_ms", "recompute_ms",
            "saved_ms", "total_ms", "effective_bandwidth_gbps", "max_abs_error", "success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--mode", default="local", choices=["local"],
                    help="local: 本地 device radix 命中(远端 write_through 被昆仑 kernel blocker 卡死)")
    ap.add_argument("--transport", default="local_radix")
    ap.add_argument("--context", default="8k,32k,128k")
    ap.add_argument("--hit", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1, help="每 shape 先发几个 throwaway 触发 JIT")
    ap.add_argument("--out", default="/root/prefill_local.csv")
    args = ap.parse_args()

    st = s_token_v32()
    print(f"[cfg] V3.2 MLA S_token/token = {st} B ({st/KiB:.1f} KiB), mode={args.mode}, base={args.base}", flush=True)
    ctxs = parse_ctx(args.context)
    hits = [float(x) for x in args.hit.replace(" ", "").split(",") if x]
    rows = []
    for N in ctxs:
        for h in hits:
            try:
                bench_one(args.base, N, h, args.page_size, args.iters, args.warmup,
                          st, rows, args.transport, args.mode)
            except Exception as e:
                print(f"  [err] N={N} h={h}: {e}", flush=True)
    write_csv(args.out, rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
