#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue5 真实 KV Tensor Transfer Microbench 

在 L1(随机 buffer) 之上, 按真实模型 config 造 **每 token KV 的 shape/dtype/分层布局**,
走同一个 mooncake store, 但**镜像 sglang hicache 的真实零拷贝路径**:
  register_buffer(整块 KV buffer) -> batch_put_from / batch_get_into (逐 page, 每 page 2 个 key)
对照一条 **naive bytes 路径**(put_batch/get_batch + python bytes), 复现 L1 发现的 GET 瓶颈,
并证明零拷贝能否让 retrieve 突破 ~1.9 GB/s。仍**不跑 attention**, 因此绕开两个昆仑算子 blocker。

真实布局参考 sglang mooncake_store.py:
  - MHA/GQA: 每 page 2 个 key ({k}, {v}); bytes_per_page = ksize_per_token * page_size
  - MLA:     每 page 1 个 key; latent = (kv_lora_rank + qk_rope_head_dim)
  - register_buffer(data_ptr, nbytes) 一次, 逐 page 给 (ptr+offset, size) 做 batch_put_from/get_into
  - batch_put_from 返回码: 每元素 0 成功; batch_get_into 返回码: 每元素 >0(读到的字节数) 成功

用法 (prefill pod 内, conda env python310_torch29_cuda):
  export PATH=/root/miniconda/envs/python310_torch29_cuda/bin:$PATH
  TRANSPORT=tcp  python3 benchmark_kv_tensor.py --model qwen3-32b --out kv_tcp.csv
  TRANSPORT=rdma python3 benchmark_kv_tensor.py --model qwen3-32b --device mlx5_3 --out kv_rdma.csv
"""
import argparse, os, time, uuid, statistics, csv, sys

GB = 1 << 30
MiB = 1 << 20
KiB = 1 << 10

# 模型 config 预设 (每 TP rank 视角): 逐层 KV, K 与 V 各一份
# S_token(整模型/rank) = 2 * L * H_kv * D_h * b   (GQA/MHA)
#                       = L * (kv_lora_rank + qk_rope) * b, 单 latent (MLA, 无独立 V)
MODELS = {
    # Qwen3-32B: L=64, H_kv=8, D_h=128, bf16; 整模型 256 KiB/token, TP8 -> 32 KiB/token/rank
    "qwen3-32b": dict(kind="mha", num_layers=64, num_kv_heads=8, head_dim=128, dtype_bytes=2),
    # DeepSeek-V3.2 (MLA): L=61, kv_lora_rank=512, qk_rope_head_dim=64, bf16; 单 latent, 无独立 V
    "deepseek-v32": dict(kind="mla", num_layers=61, kv_lora_rank=512, qk_rope=64, dtype_bytes=2),
}


def human(n):
    if n >= GB: return f"{n/GB:.1f}GB"
    if n >= MiB: return f"{n/MiB:.0f}MB"
    return f"{n}B"


def parse_ctx(s):
    out = []
    for tok in s.replace(" ", "").lower().split(","):
        if not tok: continue
        if tok.endswith("k"): out.append(int(float(tok[:-1]) * 1024))
        else: out.append(int(tok))
    return out


def s_token_per_rank(cfg, tp):
    """每 token / 每 rank 的 KV 字节数。"""
    if cfg["kind"] == "mha":
        hkv = cfg["num_kv_heads"] // tp
        assert hkv >= 1, "num_kv_heads 必须 >= tp"
        return 2 * cfg["num_layers"] * hkv * cfg["head_dim"] * cfg["dtype_bytes"]
    else:  # mla: 单 latent, 逐层, 不随 TP 切分 (MLA latent 各 rank 复制/或 tp=1 视角)
        return cfg["num_layers"] * (cfg["kv_lora_rank"] + cfg["qk_rope"]) * cfg["dtype_bytes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.environ.get("KV_IP", "192.0.2.10") + ":50051")
    ap.add_argument("--metadata", default="http://" + os.environ.get("KV_IP", "192.0.2.10") + ":18080/metadata")
    ap.add_argument("--protocol", default=os.environ.get("TRANSPORT", "tcp"))
    ap.add_argument("--device", default=os.environ.get("MC_DEVICE", ""))
    ap.add_argument("--local-hostname", default=os.environ.get("HOST_IP", ""))
    ap.add_argument("--model", default="qwen3-32b", choices=list(MODELS))
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--context", default="8k,32k,128k")
    ap.add_argument("--page-size", type=int, default=64)
    ap.add_argument("--paths", default="bytes,zerocopy")  # 逗号: bytes / zerocopy
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--local-buffer-gb", type=float, default=1.0)
    ap.add_argument("--out", default="/root/kv_tensor.csv")
    args = ap.parse_args()

    if not args.local_hostname:
        import socket
        args.local_hostname = socket.gethostbyname(socket.gethostname())

    global torch
    import torch as torch
    from mooncake.store import MooncakeDistributedStore
    store = MooncakeDistributedStore()
    ret = store.setup(args.local_hostname, args.metadata, 0,
                      int(args.local_buffer_gb * GB), args.protocol,
                      args.device, args.master)
    if ret:
        print(f"[FATAL] store.setup ret={ret}", file=sys.stderr); sys.exit(1)
    print(f"[ok] setup protocol={args.protocol} device='{args.device}' model={args.model} "
          f"tp={args.tp} local={args.local_hostname}", flush=True)

    cfg = MODELS[args.model]
    st = s_token_per_rank(cfg, args.tp)
    print(f"[cfg] S_token/rank = {st} B ({human(st)}), page={args.page_size} "
          f"-> {human(st*args.page_size)}/page", flush=True)

    ctxs = parse_ctx(args.context)
    paths = [p for p in args.paths.split(",") if p.strip()]
    rows, tag = [], uuid.uuid4().hex[:8]

    try:                                   # 清空可能的历史残留, 保证干净起点
        n = store.remove_all(); print(f"[clean] remove_all removed {n} objects", flush=True)
    except Exception as e:
        print(f"[clean] remove_all skip: {e}", flush=True)

    for N in ctxs:
        for path in paths:
            _bench(store, cfg, args, N, st, path, tag, rows)

    _write_csv(args.out, rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out}", flush=True)


def _layout(cfg, N, st, page_size):
    """返回 (num_pages, keys_per_page, key_size_bytes, kv_bytes)。
    mha: 每 page 2 key(k,v), 各 (st/2)*page; mla: 每 page 1 key, st*page。"""
    num_pages = N // page_size
    if cfg["kind"] == "mha":
        keys_per_page = 2
        key_size = (st // 2) * page_size
    else:
        keys_per_page = 1
        key_size = st * page_size
    kv_bytes = num_pages * keys_per_page * key_size
    return num_pages, keys_per_page, key_size, kv_bytes


def _median_bw(times, kv_bytes):
    if not times: return 0.0, 0.0
    med = statistics.median(times)
    bw = (kv_bytes / med / GB) if med > 0 else 0.0
    return med * 1e3, bw  # ms, GB/s


def _bench(store, cfg, args, N, st, path, tag, rows):
    ps = args.page_size
    num_pages, kpp, key_size, kv_bytes = _layout(cfg, N, st, ps)
    numel = kv_bytes // cfg["dtype_bytes"]   # bf16 元素数
    prefix = f"kv_{tag}_{args.model}_{N}_{path}"

    # 造真实 KV: 逐元素 bf16, 值域 ~ N(0,0.1) 贴近真实激活尺度; read 端清零
    torch.manual_seed(1234)
    write = (torch.randn(numel, dtype=torch.float32) * 0.1).to(torch.bfloat16).contiguous()
    read = torch.zeros(numel, dtype=torch.bfloat16).contiguous()

    # 每 iter 唯一 key (含 it): 必须如此才能测到真实 PUT 传输带宽。
    # 关键坑 (见 doc/步骤3): TCP 模式下 mooncake put/batch_put_from 对**已存在的 key 是
    #   no-op(按 key 去重, 直接跳过传输)** -> 若复用同一批 key, iter>=1 的 put 只有 ~3ms
    #   (假带宽 ~90GB/s)。故逐 iter 用唯一 key 做真插入, 并在每轮结束后 remove 释放 working
    #   set(对未被覆盖写的新 key, TCP/RDMA 下 remove 均返回 0), 避免 handle 累积耗尽。
    def page_keys(it):
        ks = []
        for p in range(num_pages):
            ks.append(f"{prefix}_{it}_{p}_k")
            if kpp == 2:
                ks.append(f"{prefix}_{it}_{p}_v")
        return ks

    put_t, get_t, max_err, ok = [], [], -1.0, 1

    if path == "zerocopy":
        # 镜像 sglang: register_buffer 整块, batch_put_from/batch_get_into 逐 key(ptr+offset,size)
        rc1 = store.register_buffer(write.data_ptr(), write.numel() * cfg["dtype_bytes"])
        rc2 = store.register_buffer(read.data_ptr(), read.numel() * cfg["dtype_bytes"])
        if rc1 or rc2:
            print(f"[warn] register_buffer rc=({rc1},{rc2}) path={path} N={N}", flush=True)
        wptrs = [write.data_ptr() + i * key_size for i in range(num_pages * kpp)]
        rptrs = [read.data_ptr() + i * key_size for i in range(num_pages * kpp)]
        sizes = [key_size] * (num_pages * kpp)
        for it in range(args.iters):
            ks = page_keys(it)
            t = time.perf_counter(); pr = store.batch_put_from(ks, wptrs, sizes)
            put_t.append(time.perf_counter() - t)
            if any(r != 0 for r in pr): ok = 0
            t = time.perf_counter(); gr = store.batch_get_into(ks, rptrs, sizes)
            get_t.append(time.perf_counter() - t)
            if any(r <= 0 for r in gr): ok = 0
            for k in ks:                       # 逐轮清理 (untimed), 释放 handle
                try: store.remove(k)
                except Exception: pass
        max_err = float((read.to(torch.float32) - write.to(torch.float32)).abs().max())
        store.unregister_buffer(write.data_ptr()); store.unregister_buffer(read.data_ptr())
    else:
        max_err, ok = _bench_bytes(store, write, page_keys, key_size,
                                   num_pages, kpp, args.iters, put_t, get_t)
    # 兜底: 清掉本 context 可能残留的全部 key (remove_all 在 TCP/RDMA 下均可用)
    try: store.remove_all()
    except Exception: pass

    put_ms, put_bw = _median_bw(put_t, kv_bytes)
    get_ms, get_bw = _median_bw(get_t, kv_bytes)
    for op, ms, bw, me in (("put", put_ms, put_bw, -1.0), ("get", get_ms, get_bw, max_err)):
        rows.append(dict(
            transport=args.protocol, model=args.model, path=path, op=op,
            context=N, tp=args.tp, page_size=ps, num_pages=num_pages,
            num_keys=num_pages * kpp, kv_bytes=kv_bytes, kv_size=human(kv_bytes),
            iters=args.iters, transfer_ms=round(ms, 3),
            effective_bandwidth_gbps=round(bw, 3),
            max_abs_error=("" if me < 0 else round(me, 6)), success=ok))
        print(f"  {args.protocol:4} {path:8} {op} N={N:<7} {human(kv_bytes):>7} "
              f"keys={num_pages*kpp:<5} BW={bw:.2f} GB/s {ms:.1f}ms "
              f"err={'' if me<0 else round(me,5)} ok={ok}", flush=True)
    del write, read


def _bench_bytes(store, write, page_keys, key_size, num_pages, kpp, iters, put_t, get_t):
    """naive bytes 路径: put_batch/get_batch, 复现 L1 的 python bytes 拷贝瓶颈。"""
    u8 = write.view(torch.uint8).contiguous()
    mv = memoryview(u8.numpy())
    vals = [bytes(mv[i * key_size:(i + 1) * key_size]) for i in range(num_pages * kpp)]
    max_err, ok = -1.0, 1
    for it in range(iters):
        ks = page_keys(it)
        t = time.perf_counter(); rc = store.put_batch(ks, vals)
        put_t.append(time.perf_counter() - t)
        if rc != 0: ok = 0
        t = time.perf_counter(); got = store.get_batch(ks)
        get_t.append(time.perf_counter() - t)
        if it == iters - 1:
            bad = sum(1 for g in got if (g is None or len(g) != key_size))
            if bad: ok = 0
            max_err = 0.0 if (not bad and all(bytes(got[i]) == vals[i]
                              for i in range(len(vals)))) else 1.0
        for k in ks:
            try: store.remove(k)
            except Exception: pass
    return max_err, ok


def _write_csv(path, rows):
    if not rows: return
    cols = ["transport", "model", "path", "op", "context", "tp", "page_size",
            "num_pages", "num_keys", "kv_bytes", "kv_size", "iters", "transfer_ms",
            "effective_bandwidth_gbps", "max_abs_error", "success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)


if __name__ == "__main__":
    main()
