#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Issue5 步骤2 / Layer1 —— Mooncake primitive microbench (不进模型)

直接用 mooncake python client 连已起的远端 store (kvstore 节点), put/get 不同大小 buffer,
测传输层本身的带宽/延迟。TCP vs RDMA 用 TRANSPORT 环境变量或 --protocol 切换。

用法 (在 prefill pod 内, conda env python310_torch29_cuda):
  export PATH=/root/miniconda/envs/python310_torch29_cuda/bin:$PATH
  TRANSPORT=tcp  python3 benchmark_transfer.py --out /root/transfer_tcp.csv
  TRANSPORT=rdma python3 benchmark_transfer.py --out /root/transfer_rdma.csv --device eth1

setup(local_hostname, metadata_server, global_segment_size, local_buffer_size,
      protocol, device_name, master_server_address)  # 7 位置参数, 见 sglang mooncake_store.py
"""
import argparse, os, time, uuid, statistics, csv, sys
from concurrent.futures import ThreadPoolExecutor

GB = 1 << 30
MB = 1 << 20


def human(n):
    if n >= GB: return f"{n/GB:.0f}GB"
    if n >= MB: return f"{n/MB:.0f}MB"
    return f"{n}B"


def parse_sizes(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        if not tok: continue
        if tok.endswith("gb"): out.append(int(float(tok[:-2]) * GB))
        elif tok.endswith("mb"): out.append(int(float(tok[:-2]) * MB))
        elif tok.endswith("kb"): out.append(int(float(tok[:-2]) * 1024))
        else: out.append(int(tok))
    return out


def pctl(vals, p):
    if not vals: return 0.0
    xs = sorted(vals)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.environ.get("KV_IP", "192.0.2.10") + ":50051")
    ap.add_argument("--metadata", default="http://" + os.environ.get("KV_IP", "192.0.2.10") + ":18080/metadata")
    ap.add_argument("--protocol", default=os.environ.get("TRANSPORT", "tcp"))
    ap.add_argument("--device", default=os.environ.get("MC_DEVICE", ""))  # rdma NIC 名, tcp 时留空
    ap.add_argument("--local-hostname", default=os.environ.get("HOST_IP", ""))
    ap.add_argument("--sizes", default="64mb,256mb,1gb")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--local-buffer-gb", type=float, default=5.0)
    ap.add_argument("--mode", default="simple", choices=["simple", "zcget", "zcput"],
                    help="simple=原有 put+bytes-get; zcget=零拷贝 batch_get_into 取回; "
                         "zcput=零拷贝 batch_put_from 写入 (均镜像 hicache 真实路径)")
    ap.add_argument("--out", default="/root/transfer.csv")
    args = ap.parse_args()

    if not args.local_hostname:
        import socket
        args.local_hostname = socket.gethostbyname(socket.gethostname())

    from mooncake.store import MooncakeDistributedStore
    store = MooncakeDistributedStore()
    # global_segment_size=0: client 不贡献存储, 所有数据落到远端 kvstore 段 -> 真跨网传输
    ret = store.setup(args.local_hostname, args.metadata, 0,
                      int(args.local_buffer_gb * GB), args.protocol,
                      args.device, args.master)
    if ret:
        print(f"[FATAL] store.setup ret={ret}", file=sys.stderr); sys.exit(1)
    print(f"[ok] setup protocol={args.protocol} device='{args.device}' "
          f"local={args.local_hostname} master={args.master}", flush=True)

    sizes = parse_sizes(args.sizes)
    concs = [int(x) for x in args.concurrency.split(",") if x.strip()]
    rows = []
    run_tag = uuid.uuid4().hex[:8]

    for size in sizes:
        payload = os.urandom(min(size, 64 * MB))
        # 大 value 用重复拼接避免 os.urandom 太慢, 但仍保持 size 字节
        if size > len(payload):
            payload = (payload * (size // len(payload) + 1))[:size]
        for conc in concs:
            if args.mode == "zcget":
                _bench_zc_get(store, size, conc, args.iters, payload, run_tag, rows, args.protocol)
            elif args.mode == "zcput":
                _bench_zc_put(store, size, conc, args.iters, payload, run_tag, rows, args.protocol)
            else:
                _bench_one(store, size, conc, args.iters, payload, run_tag, rows, args.protocol)

    _write_csv(args.out, rows)
    print(f"[done] wrote {len(rows)} rows -> {args.out}", flush=True)
    try: store.close()
    except Exception: pass


def _bench_one(store, size, conc, iters, payload, run_tag, rows, protocol):
    keys = [f"bench_{run_tag}_{human(size)}_{conc}_{i}" for i in range(iters)]

    def do_put(i):
        t = time.perf_counter()
        rc = store.put(keys[i], payload)
        return (time.perf_counter() - t, rc)

    def do_get(i):
        t = time.perf_counter()
        v = store.get(keys[i])
        dt = time.perf_counter() - t
        return (dt, 0 if (v is not None and len(v) == size) else 1)

    # ---- PUT ----
    put_lat, put_ok, put_wall = _run_parallel(do_put, iters, conc)
    # ---- GET ----
    get_lat, get_ok, get_wall = _run_parallel(do_get, iters, conc)

    for op, lats, oks, wall in (("put", put_lat, put_ok, put_wall),
                                 ("get", get_lat, get_ok, get_wall)):
        # 有效聚合带宽 = 总字节 / 实测批次墙钟时间 (并发下反映真实吞吐)
        total_bytes = size * len(lats)
        ok = int(sum(oks) == 0)
        bw = (total_bytes / wall / GB) if (wall > 0 and ok) else 0.0
        row = dict(transport=protocol, op=op, size_bytes=size, size=human(size),
                   concurrency=conc, iters=len(lats), total_bytes=total_bytes,
                   wall_s=round(wall, 4), bandwidth_gbps=round(bw, 3),
                   lat_p50_ms=round(pctl(lats, 50) * 1e3, 3),
                   lat_p95_ms=round(pctl(lats, 95) * 1e3, 3),
                   lat_p99_ms=round(pctl(lats, 99) * 1e3, 3),
                   lat_mean_ms=round(statistics.mean(lats) * 1e3, 3) if lats else 0,
                   success=ok)
        rows.append(row)
        print(f"  {protocol:4} {op} {human(size):>6} c={conc:<2} "
              f"BW={row['bandwidth_gbps']:.2f} GB/s  p50={row['lat_p50_ms']:.1f}ms "
              f"p99={row['lat_p99_ms']:.1f}ms ok={row['success']}", flush=True)

    # 清理 keys
    for k in keys:
        try: store.remove(k)
        except Exception: pass


def _bench_zc_put(store, size, conc, iters, payload, run_tag, rows, protocol):
    """零拷贝 PUT: 镜像 sglang hicache 的 register_buffer + batch_put_from 真实 write_through 路径。
    对照 _bench_one 的 simple bytes put(store.put 传 python bytes), 这里一次 register_buffer 整块
    write 缓冲, 一批 batch_put_from(keys, ptrs, sizes) 从预注册张量直 DMA 出去, 无 python 中间对象。
    并发 = 单批次内并行写 conc 个 buffer, BW = conc*size / 批次中位墙钟时间。
    ⚠️ 关键坑(见 doc/步骤3): TCP 下 put/batch_put_from 对**已存在的 key 是 no-op(按 key 去重,
    直接跳过传输)** -> 必须每 iter 用唯一 key(含 it) 做真插入, 否则 iter>=1 只花 ~3ms(假带宽)。"""
    global torch
    import torch as torch
    # 整块 write 缓冲 conc*size 字节, 用 payload 填充(避免全零), 注册一次
    write = torch.frombuffer(bytearray(payload * conc), dtype=torch.uint8).clone().contiguous()
    rc = store.register_buffer(write.data_ptr(), conc * size)
    if rc:
        print(f"[warn] register_buffer rc={rc} zcput size={human(size)} c={conc}", flush=True)
    wptrs = [write.data_ptr() + i * size for i in range(conc)]
    sizes = [size] * conc
    put_t, ok = [], 1
    for it in range(iters):
        keys = [f"zcput_{run_tag}_{human(size)}_{conc}_{it}_{i}" for i in range(conc)]
        t = time.perf_counter()
        pr = store.batch_put_from(keys, wptrs, sizes)
        put_t.append(time.perf_counter() - t)
        if any(r != 0 for r in pr): ok = 0
        for k in keys:                      # 逐轮清理(untimed), 释放 handle 并避免下轮去重
            try: store.remove(k)
            except Exception: pass
    med = statistics.median(put_t) if put_t else 0.0
    total_bytes = size * conc
    bw = (total_bytes / med / GB) if (med > 0 and ok) else 0.0
    row = dict(transport=protocol, op="put_zc", size_bytes=size, size=human(size),
               concurrency=conc, iters=len(put_t), total_bytes=total_bytes,
               wall_s=round(med, 4), bandwidth_gbps=round(bw, 3),
               lat_p50_ms=round(pctl(put_t, 50) * 1e3, 3),
               lat_p95_ms=round(pctl(put_t, 95) * 1e3, 3),
               lat_p99_ms=round(pctl(put_t, 99) * 1e3, 3),
               lat_mean_ms=round(statistics.mean(put_t) * 1e3, 3) if put_t else 0,
               success=ok)
    rows.append(row)
    print(f"  {protocol:4} put_zc {human(size):>6} c={conc:<2} "
          f"BW={row['bandwidth_gbps']:.2f} GB/s  p50={row['lat_p50_ms']:.1f}ms "
          f"p99={row['lat_p99_ms']:.1f}ms ok={row['success']}", flush=True)
    store.unregister_buffer(write.data_ptr())
    try: store.remove_all()
    except Exception: pass


def _bench_zc_get(store, size, conc, iters, payload, run_tag, rows, protocol):
    """零拷贝 GET: 镜像 sglang hicache 的 register_buffer + batch_get_into 真实 retrieve 路径。
    对照 _bench_one 的 simple bytes get(每 key 新建 python bytes + memcpy + GIL, 卡 ~1.9GB/s),
    这里一次 register_buffer 整块 read 缓冲, 逐 key 直 DMA 进预注册张量, 无 python 中间对象。
    并发用「一次 batch_get_into 取 conc 个 key」表达(与 simple 的 ThreadPoolExecutor 并发对齐语义:
    同一批次内并行取 conc 个 buffer), BW = conc*size / 批次中位墙钟时间。"""
    global torch
    import torch as torch
    keys = [f"zcget_{run_tag}_{human(size)}_{conc}_{i}" for i in range(conc)]
    # 预置数据 (untimed setup): 每 key 存 size 字节 payload
    for k in keys:
        store.put(k, payload)
    # 整块 read 缓冲 conc*size 字节, 注册一次
    read = torch.zeros(conc * size, dtype=torch.uint8).contiguous()
    rc = store.register_buffer(read.data_ptr(), conc * size)
    if rc:
        print(f"[warn] register_buffer rc={rc} zcget size={human(size)} c={conc}", flush=True)
    rptrs = [read.data_ptr() + i * size for i in range(conc)]
    sizes = [size] * conc
    get_t, ok = [], 1
    for _ in range(iters):
        t = time.perf_counter()
        gr = store.batch_get_into(keys, rptrs, sizes)
        get_t.append(time.perf_counter() - t)
        if any(r <= 0 for r in gr): ok = 0
    med = statistics.median(get_t) if get_t else 0.0
    total_bytes = size * conc
    bw = (total_bytes / med / GB) if (med > 0 and ok) else 0.0
    row = dict(transport=protocol, op="get_zc", size_bytes=size, size=human(size),
               concurrency=conc, iters=len(get_t), total_bytes=total_bytes,
               wall_s=round(med, 4), bandwidth_gbps=round(bw, 3),
               lat_p50_ms=round(pctl(get_t, 50) * 1e3, 3),
               lat_p95_ms=round(pctl(get_t, 95) * 1e3, 3),
               lat_p99_ms=round(pctl(get_t, 99) * 1e3, 3),
               lat_mean_ms=round(statistics.mean(get_t) * 1e3, 3) if get_t else 0,
               success=ok)
    rows.append(row)
    print(f"  {protocol:4} get_zc {human(size):>6} c={conc:<2} "
          f"BW={row['bandwidth_gbps']:.2f} GB/s  p50={row['lat_p50_ms']:.1f}ms "
          f"p99={row['lat_p99_ms']:.1f}ms ok={row['success']}", flush=True)
    store.unregister_buffer(read.data_ptr())
    for k in keys:
        try: store.remove(k)
        except Exception: pass


def _run_parallel(fn, iters, conc):
    lats, oks = [], []
    t0 = time.perf_counter()
    if conc <= 1:
        for i in range(iters):
            dt, rc = fn(i); lats.append(dt); oks.append(rc)
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            for dt, rc in ex.map(fn, range(iters)):
                lats.append(dt); oks.append(rc)
    wall = time.perf_counter() - t0
    return lats, oks, wall


def _write_csv(path, rows):
    if not rows: return
    cols = ["transport", "op", "size", "size_bytes", "concurrency", "iters",
            "total_bytes", "wall_s", "bandwidth_gbps", "lat_mean_ms",
            "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "success"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)


if __name__ == "__main__":
    main()
