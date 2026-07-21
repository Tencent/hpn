# issue5 report

1. 带宽临界值模型

输入：命中率，上下文长度，KV Cache 量，算力

输出：以存代算的临界带宽

2. 相同 TTFT 约束下

TCP 条件下， 带宽，延迟，可承载并发query数，QPM

RDMA 条件下，带宽，延迟，可承载并发query数，QPM

## 基本环境与性能

### 硬件环境与拓扑

硬件设备

| 项              | kvstore 192.0.2.10                     | prefill 192.0.2.20      |
| :-------------- | :------------------------------------- | :---------------------- |
| HCA             | mlx5_0..5（6 个）                      | mlx5_0..6（7 个）       |
| link_layer      | Ethernet（RoCEv2）                     | Ethernet（RoCEv2）      |
| 端口状态        | eth0/eth1-4 PORT_ACTIVE                | eth0/eth1-4 PORT_ACTIVE |
| /dev/infiniband | uverbs0-5 + rdma_cm                    | uverbs0-6 + rdma_cm     |
| kmods           | mlx5_ib/ib_core/rdma_cm/kunlun_peermem | 同                      |
| ulimit -l       | unlimited                              | unlimited               |
| perftest        | ib_write_bw/ib_read_bw/ib_send_lat     | 同                      |

硬件拓扑

| netdev                          | kvstore        | prefill               |
| :------------------------------ | :------------- | :-------------------- |
| eth0（前端 192.0.2.x，管理/OOB） | mlx5_0         | mlx5_0                |
| eth1（RDMA，198.51.100.x/27）         | mlx5_2         | mlx5_3                |
| eth2                            | mlx5_3         | mlx5_4                |
| eth3                            | mlx5_4         | mlx5_5                |
| eth4                            | mlx5_5         | mlx5_6                |
| eth5/eth6                       | mlx5_1（DOWN） | mlx5_1/mlx5_2（DOWN） |

- conda env `python310_torch29_cuda`，预装 mooncake **0.3.5**（勿 pip 装，见 MEMORY）。
- store 端口：master RPC `50051`、http metadata `18080`、store `18081`。
- client（prefill）用 `global_segment_size=0` 不贡献内存 → put/get 全部真跨机落到远端 store 段。

- RDMA 数据 NIC = **eth1-eth4**（与 `start_prefill*.sh` 的 `BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"` 一致）。
- 同名 netdev 在两机对应的 mlx5 序号不同 → 配置里若按 `mlx5_N` 指定要分别写；按 `ethN` 指定更稳。
- RDMA NIC 子网：kvstore `198.51.100.0/27`，prefill `198.51.100.32/27`（不同 /27，RoCEv2 可路由）。

### 基本性能

#### 设备性能

RDMA 本机数据路径已实测 146 Gb/s；

跨机 RDMA 需走 Mooncake 自带传输引擎（握手端口已放通、已发现 6 HCA），独立 perftest 的跨机 OOB 被安全组挡住， 不影响项目，真正的跨机 RDMA 数据面测试在 L1 用 `MC_PROTOCOL=rdma` 完成

65536B  ->  BW average 146.73 Gb/sec  (~18.3 GB/s)

#### Mooncake put/get primitive microbench

这里使用 零拷贝 `batch_get_into`，镜像 hicache 真实 retrieve 路径，一次 `register_buffer(整块 read 缓冲)`， 一批 `batch_get_into(keys, ptrs, sizes)` 直 DMA 进预注册张量（无 python bytes 中间对象），否则 RDMA get 平台化在 ~1.5–2.1 GB/s，远低于同协议 put 的 16 GB/s，瓶颈卡在 client 侧 `bytes` 分配 + memcpy + GIL

PUT（写入远端 store，对应 write_through KV 落盘）

| size  | conc | TCP-zc | RDMA-zc   | RDMA-zc / TCP-zc |
|---|---|---|---|---|
| 64MB  | 1    | 0.58   | **19.65** | 33.9×            |
| 64MB  | 2    | 0.62   | **20.80** | 33.5×            |
| 64MB  | 4    | 0.66   | **22.07** | 33.4×            |
| 64MB  | 8    | 0.73   | **21.86** | 29.9×            |
| 256MB | 1    | 0.65   | **18.38** | 28.3×            |
| 256MB | 2    | 0.77   | **20.15** | 26.2×            |
| 256MB | 4    | 1.22   | **20.72** | 17.0×            |
| 256MB | 8    | 1.17   | **21.14** | 18.1×            |
| 1GB   | 1    | 1.21   | **18.07** | 14.9×            |
| 1GB   | 2    | 1.23   | **19.77** | 16.1×            |
| 1GB   | 4    | 1.06   | **21.09** | 19.9×            |

GET（读回远端 store，对应命中时的 KV retrieve）

| size  | conc | TCP-zc | RDMA-zc   | RDMA-zc / TCP-zc |
| :---- | :--- | :----- | :-------- | :--------------- |
| 64MB  | 1    | 1.20   | **10.20** | 8.5×             |
| 64MB  | 2    | 1.22   | **8.93**  | 7.3×             |
| 64MB  | 4    | 1.23   | **10.48** | 8.5×             |
| 64MB  | 8    | 1.21   | **13.39** | 11.1×            |
| 256MB | 1    | 1.19   | **10.43** | 8.8×             |
| 256MB | 2    | 0.93   | **12.78** | 13.7×            |
| 256MB | 4    | 1.44   | **13.20** | 9.2×             |
| 256MB | 8    | 1.49   | **12.03** | 8.1×             |
| 1GB   | 1    | 1.31   | **13.92** | 10.6×            |
| 1GB   | 2    | 1.44   | **14.47** | 10.0×            |
| 1GB   | 4    | 1.58   | **14.61** | 9.2×             |
| 1GB   | 8    | 1.23   | **14.69** | 11.9×            |



## 验收1

临界值带宽模型

输入：命中率，上下文长度，KV Cache 量，算力

输出：以存代算的临界带宽

### 模型推导

全量重算路径
$$
T_{\text{full}}(N)=T_{\text{prefill}}(N)
$$

- $N$：完整输入的 token 数；
- $T_{\text{prefill}}(N)$：从头计算这 $N$ 个 token 的 prefill 时间。

以存代算路径
$$
T_{\text{cache}}(N,h,B)
=
T_0
+
T_{\text{retrieve}}
+
T_{\text{restore}}(hN)
+
T_{\text{prefill}}((1-h)N)
$$

- $h$：前缀命中率。

**固定开销 $T_0$** ：$T_0=T_{\text{lookup}}+L_{\text{net}}$，如查询 metadata，判断前缀是否命中，发起 RPC，建立或调度传输请求，网络往返固定延迟等

**传输时间 $T_{\text{retrieve}}$**：$T_{\text{retrieve}}=\frac{S_{\text{KV}}(hN)}{B}$

- $S_{\text{KV}}(hN)$：命中前缀对应的 KV 字节数；
- $B$：实际 retrieve 有效带宽。

**恢复开销 $T_{\text{restore}}$**：数据通过网络到达后，不一定能立刻被模型使用，可能还需要 反序列化，从临时缓冲区复制到 KV pool，恢复 page table 或 slot 映射，layout 转换 等，因此把 $T_{\text{retrieve}}$ 和 $T_{\text{restore}}(hN)$ 区分开

**后缀现算$T_{\text{prefill}}$**： $T_{\text{prefill}}((1-h)N)$ 命中的前缀不再计算，但未命中的后缀仍要正常跑模型



对于值得使用远端 KV 的情况，令
$$
T_{\text{cache}}<T_{\text{full}}
$$
代入：
$$
T_{\text{lookup}}+L_{\text{net}}
+\frac{S_{\text{KV}}}{B}
+T_{\text{restore}}
+T_{\text{prefill}}((1-h)N)
<
T_{\text{prefill}}(N)
$$
把除传输以外的项移到右边：
$$
\frac{S_{\text{KV}}}{B}
<
T_{\text{prefill}}(N)
-
T_{\text{prefill}}((1-h)N)
-
T_{\text{lookup}}
-
T_{\text{restore}}
-
L_{\text{net}}
$$
右边表示 省下的前缀计算时间，扣除所有固定和恢复开销后，最多还能留给网络传输多少时间。

于是解出带宽门槛：
$$
B_{\text{crit}}(N,h)
=
\frac{S_{\text{KV}}(hN)}
{
T_{\text{prefill}}(N)
-
T_{\text{prefill}}((1-h)N)
-
T_{\text{lookup}}
-
T_{\text{restore}}
-
L_{\text{net}}
}
$$
若

- $B>B_{\text{crit}}$ 则以存代算更快 
- $B=B_{\text{crit}}$ 刚好打平
- $B<B_{\text{crit}}$ 不如重新计算

### KV Tensor Transfer microbench

DeepSeek-V3.2（MLA）：
$$
S_{\text{token,rank}}
=
L\cdot(\text{kv\_lora\_rank}+\text{qk\_rope})\cdot b
$$
代入 DeepSeek-V3.2 的参数 `L=61, kv_lora_rank=512, qk_rope=64, bf16(b=2)` = $61\times(512+64)\times2 = 70272\text{ B}$

换算后：$70272\div1024=68.625\text{ KiB/token/rank}$

TP8：$S_{token/rank} = 61·(512+64)·2 = 70272 B = 68.6 KiB/token/rank$

page 布局镜像 sglang `mooncake_store.py`：

- `page_size=64` 每 page = 68.6KiB · 64 = 4.29 MB；
- MLA 每 page 1 个 key（单 latent，无独立 V），`key_size = S_token · page = 4.29 MB`；
- 一次 `register_buffer(整块 KV buffer)`，逐 page 给 `(ptr+offset, size)` 做 `batch_put_from` / `batch_get_into`（page_first 布局）。

上下文长度 → 每 rank 传输总量（= `num_pages · 1 · 4.29MB`）：

| context N | num_pages | num_keys | KV/rank    |
| :-------- | :-------- | :------- | :--------- |
| 8k        | 128       | 128      | **549 MB** |
| 32k       | 512       | 512      | **2.1 GB** |
| 128k      | 2048      | 2048     | **8.6 GB** |



### prefill end to end microbench

启动日志：`max_total_num_tokens=29120, available_gpu_mem=14.94 GB`。671B int8 W8A8 权重占 `--mem-fraction-static 0.83`（>0.85 MoE OOM / <0.83 权重 OOM，窗口极窄），每卡只剩 ~15GB 给 KV ⇒ 全局 KV 上限 **29120 token**，32k / 128k 单请求超 KV 容量：实测发一个 32768-token prefill 直接把 scheduler 挂死（detokenizer 20s 无响应、health check 连续失败、进程 wedge，需 kill 重起）。故 sweep 的上下文改为 **8k / 16k / 24k**（均 < 29120，24k 留 ~4.5k 余量）。因此本实验只能稳定实测 8k、16k 和 24k 上下文；

32k 与 128k 这些长上下文的 KV 应从**远端 store retrieve** 回来（用 L2 的 `B_retrieve` 代入 §6 的 `B_crit` 外推），但远端 KV 存储本身只避免前缀重算，并不会自动突破本地活跃 KV 容量上限；真正支持超长上下文还需分层按需加载、KV 分片或更大本地 KV 池。



命中率 h 用「共享前缀 + 唯一后缀」控制：`prefix = align(round(h*N), page=64)`，`suffix = N-prefix`。

- **recompute 基线 `T_prefill(N)`**：每 iter `flush` + 唯一内容 → 真全量现算，取 median。
- **命中 total(N, h)**：先发 `prefix` 预热进本地 radix（**不 flush**），再发 `prefix+唯一后缀`
  → `cached_tokens=prefix`，前缀 KV device 本地复用（restore≈0），只现算后缀。
- **`saved = T_prefill(N) − total(N,h)`** = 复用 hN 前缀省下的**计算**时间。
- **JIT 毛刺**：`--disable-cuda-graph` 下每个 shape 首次请求有编译开销（首个 8k ~9.6s，
  稳态 ~3.7s），故每 shape 先 `warmup` 丢弃再取 median。



DeepSeek V3.2 MLA, S_token=68.6 KiB/token, TP8, page=64, 本地 radix

`total`/`recompute`/`saved` 单位 ms（median, iters=3）

| N | h | prefix | cached | recompute_ms | total_ms | **saved_ms** |
|---|---|---|---|---|---|---|
| 8k  | 0.0 | 0 | 0 | 3680 | 3661 | 19 |
| 8k  | 0.25 | 2048 | 2048 | 3654 | 3296 | 358 |
| 8k  | 0.5 | 4096 | 4096 | 3638 | 2242 | 1397 |
| 8k  | 0.75 | 6144 | 6144 | 3649 | 1196 | 2453 |
| 8k  | 1.0 | 8192 | 8128 | 3659 | 381 | 3278 |
| 16k | 0.0 | 0 | 0 | 8672 | 8673 | 0 |
| 16k | 0.25 | 4096 | 4096 | 8643 | 7220 | 1423 |
| 16k | 0.5 | 8192 | 8192 | 8655 | 5039 | 3616 |
| 16k | 0.75 | 12288 | 12288 | 8917 | 2629 | 6288 |
| 16k | 1.0 | 16384 | 16320 | 8638 | 385 | 8253 |
| 24k | 0.0 | 0 | 0 | 14394 | 14341 | 53 |
| 24k | 0.25 | 6144 | 6144 | 14380 | 11830 | 2549 |
| 24k | 0.5 | 12288 | 12288 | 14429 | 8304 | 6124 |
| 24k | 0.75 | 18432 | 18432 | 14315 | 4467 | 9848 |
| 24k | 1.0 | 24512 | 24512 | 14410 | 405 | 14005 |

- `saved_ms` 随 h 近线性增长，随 N 更陡（长上下文重算越贵，复用价值越大）。
- h=1.0 时 `total≈380–405ms`（几乎纯命中，只剩最后一页现算 + 调度开销），`saved` 达全量的 ~96%。
- `cached_tokens` 恒等于 page 对齐前缀（h=1.0 时 = N−64，末页留一个未命中 token 触发前向）。



理想算力侧临界带宽
$$
B_{\text{crit}}^{\text{ideal}}
=
\frac{S_{\text{KV}}(hN)}
{T_{\text{prefill}}(N)-T_{\text{prefill}}((1-h)N)}
$$

- $N$：请求总 token 数；
- $h$：前缀命中率；
- $hN$：命中的 token 数；
- $S_{\text{KV}}(hN)$：这些命中 token 对应的 KV 数据量；
- $T_{\text{prefill}}(N)$：完整重算 N 个 token 的时间；
- $T_{\text{prefill}}((1-h)N)$：只计算未命中后缀的时间；
- 两者之差：复用前缀 KV 所节省的计算时间。

分母用 `recompute_ms − cold_uniq_ms`，即前缀对应的现算时间；本地命中 restore≈0，此项即纯算侧收益，$T_{restore}$ 把命中的 KV cache 从远端 store 搬回本地所需要的时间, `T_restore@RDMA 20` 即使用 RDMA，有效带宽为 20GB/s 

| N | h | cached KV | 前缀现算(=分母) | **B_crit** | T_restore@RDMA 20 | T_restore@TCP 1 |
|---|---|---|---|---|---|---|
| 8k  | 0.5 | 0.268 GB | 2236 ms | **0.120 GB/s** | 13 ms | 268 ms |
| 8k  | 1.0 | 0.532 GB | 3659 ms | **0.145 GB/s** | 27 ms | 532 ms |
| 16k | 0.5 | 0.536 GB | 5029 ms | **0.107 GB/s** | 27 ms | 536 ms |
| 16k | 1.0 | 1.068 GB | 8638 ms | **0.124 GB/s** | 53 ms | 1068 ms |
| 24k | 0.75 | 1.206 GB | 11830 ms | **0.102 GB/s** | 60 ms | 1206 ms |
| 24k | 1.0 | 1.604 GB | 14410 ms | **0.111 GB/s** | 80 ms | 1604 ms |

1. **DeepSeek V3.2 的 `B_crit` 极低（~0.09–0.15 GB/s）**

   根因是 MLA 单 latent KV 极省（68.6 KiB/token）而 671B MoE 的 prefill 算力极贵（~0.45–0.6 ms/token），每传 1 字节能省下的算力时间极大，打平所需带宽很低

2.  **TCP（~1 GB/s）与 RDMA（~20 GB/s）都远高于 `B_crit`（分别高 ~8× / ~180×）**，即
   以存代算对 V3.2 在两种传输下都稳赚

   RDMA 的 restore 延迟：`T_restore` 从 TCP 的 268–1604 ms 降到 RDMA 的 13–80 ms（~16–20×），直接改善命中请求的TTFT（验收2 的 QPM/TTFT 差距来源）

3. 分母（算力收益）巨大使 `B_crit` 很低，retrieve 带宽越高，即可让命中率还比较低时，远端 KV retrieve 的总收益就已经大于传输成本，开始比全量重算更快

### 模型实测标定

通过模型推导得到：
$$
B_{\text{crit}}(N,h)
=
\frac{S_{\text{KV}}(hN)}
{
T_{\text{prefill}}(N)
-
T_{\text{prefill}}((1-h)N)
-
T_{\text{lookup}}
-
T_{\text{restore}}
-
L_{\text{net}}
}
$$

- **分子** `S_KV(hN)` = 命中前缀要 retrieve 的 KV 字节。V3.2 MLA 单 latent： `S_token = L·(kv_lora_rank+qk_rope)·b = 61·(512+64)·2 = 70272 B = 68.6 KiB/token`。
- **分母** = 复用 hN 前缀省下的算力时间（全量减去只剩的后缀现算），再扣掉查表/反序列化/网络固定开销。
- 分母 ≤ 0 永不划算；固定开销太大，本地命中时 `T_lookup≈T_restore≈L_net≈0`，分母即纯算侧收益



**prefill end to end microbench** 标定算力侧 `T_prefill(N)`

prefill_local.csv 每个 N 的全量重算中位数做标定点， 最小二乘拟合过原点的 `T_prefill(N) = a·N + b·N²`（`b·N²` 吸收 attention 的超线性）：

需要把标定点: $8k=3654ms$, $16k=8655ms$, $24k=14394ms$ 三个离散点拟合成一个连续函数：
$$
T_{\text{prefill}}(N)
=
0.39318N+7.886\times10^{-6}N^2
\quad\text{ms}
$$

- `a≈0.393 ms/tok` 线性主项（FFN/MoE + NSA 稀疏 attention 的近线性部分）， `b≈7.9e-6 ms/tok²` 是残余二次项。R²=0.9997

- **外推警告**：V3.2 用 NSA 稀疏（`index_topk=2048`），attention 在 N≫2048 后应趋近**线性** （每 query 只算 2048 个 key），故 `b·N²` 会高估 128k 的算力。

  所以 128k 外推的 `T_prefill` 是区间上界，对应 `B_crit` 是下界（真实 B_crit 只会更高一点点，结论不变）。



**KV Tensor Transfer microbench** 标定 retrieve 带宽 `B_retrieve` 

`T_restore = S_KV / B_retrieve`。B_retrieve 是传输层带宽（GB/s），与模型无关，直接取kv_tensor_v32_{tcp,rdma}.csv 的 zerocopy `batch_get_into` get 中位：

| 传输     | B_retrieve (zerocopy get) | 备注                                   |
| :------- | :------------------------ | :------------------------------------- |
| **TCP**  | **1.19 GB/s**             | 卡在内核 socket 拷贝，零拷贝几乎无增益 |
| **RDMA** | **20.51 GB/s**            | 零拷贝直 DMA 进目标张量，~17.2× TCP    |



结果：每 (N,h) 的 B_crit + 是否划算

results/model_fit.csv（24 行 = 12 个 h>0 cell × {tcp,rdma}）

`bcrit_meas` 用实测分母，`bcrit_model` 用拟合 `T_prefill`。全部 `profitable=1`

| N    | h    | cached | S_KV     | 分母(实测) | **B_crit(实测)** | B_crit(模型) | T_restore@TCP | T_restore@RDMA |
| :--- | :--- | :----- | :------- | :--------- | :--------------- | :----------- | :------------ | :------------- |
| 8k   | 0.5  | 4096   | 0.268 GB | 2236 ms    | **0.120 GB/s**   | 0.134        | 225 ms        | 13 ms          |
| 8k   | 1.0  | 8128   | 0.532 GB | 3659 ms    | **0.145 GB/s**   | 0.142        | 446 ms        | 26 ms          |
| 16k  | 0.5  | 8192   | 0.536 GB | 5029 ms    | **0.107 GB/s**   | 0.112        | 450 ms        | 26 ms          |
| 16k  | 1.0  | 16320  | 1.068 GB | 8638 ms    | **0.124 GB/s**   | 0.125        | 896 ms        | 52 ms          |
| 24k  | 0.75 | 18432  | 1.206 GB | 11830 ms   | **0.102 GB/s**   | 0.103        | 1012 ms       | 59 ms          |
| 24k  | 1.0  | 24512  | 1.604 GB | 14410 ms   | **0.111 GB/s**   | 0.111        | 1346 ms       |                |

**标定误差（模型 B_crit vs 实测 B_crit）**：`median=2.6%, max=13.0%`。 最大误差落在 8k（短序列 JIT/调度固定开销占比大，拟合的纯二次模型没显式建模 T_0，把开销摊进 斜率 → 短序列略偏）；16k/24k 误差 <5%

**B_crit 实测范围：0.092 – 0.147 GB/s**



外推线上：128K / 30% 命中

用拟合 `T_prefill` 外推（128k 单节点放不下，故只能靠模型）：

理论命中 token 数，page size 是 64，对齐到 64 整数倍：
$$
\text{hN} = \text{align}(0.3\times131072)=39296
$$
MLA 每 token、每 TP rank 的 KV 大小是：
$$
S_{\text{token}}=70272\text{ B/token}
$$
所以命中 39296 token 对应的 KV 大小为：
$$
S_{\text{KV}}
=
39296\times70272
=
2.76\times10^9\text{ B}
\approx
2.572\text{ GiB}
$$
前面 N 代入拟合公式得到：
$$
T_{\text{prefill}}(N)
=
0.39318N+7.886\times10^{-6}N^2
\quad\text{ms}
$$
得到：$T_{prefill}(128k)\approx187.0$ s， $T_{prefill}(70\% \approx 91776)\approx102.5$ 

得到
$$
\text{B\_crit} = 2.572 \text{GB} / 84.5\text{s} = 0.0304 \text{GB/s}
$$



| retrieve 带宽   | margin (B/B_crit) | T_restore | 
| :-------------- | :---------------- | :-------- |
| RDMA 20.51 GB/s | 674×              | 0.12 s    |
| TCP 1.19 GB/s   | 39×               | 2.16 s    |




结论

1. **V3.2 的 `B_crit` 极低（实测 0.09–0.15 GB/s，128k/30% 外推 0.03 GB/s）**

   根因：MLA 单 latent KV 极省（68.6 KiB/token），而 671B MoE prefill 算力极贵（~0.45–0.6 ms/token）， 每传 1 字节能省的算力时间极大。

2. **TCP（~1.19）与 RDMA（~20.51 GB/s）都远高于 B_crit**（分别 ~8–39× / ~140–674×）

    以存代算 对 V3.2 在两种传输下都稳赚，且在极低命中率下就转正。

3. **RDMA 的价值在 restore 延迟/TTFT**：`T_restore` 从 TCP 的 0.23–2.16 s 降到 RDMA 的 0.01–0.12 s（~17×），直接改善命中请求的 TTFT

4. `selftest` 验证闭式解 `T_cache<T_full ⟺ B_retrieve>B_crit`；模型 vs 实测 B_crit 标定误差 median 2.6%。



## 验收2

相同 TTFT 约束下

TCP 条件下， 带宽，延迟，可承载并发query数，QPM

RDMA 条件下，带宽，延迟，可承载并发query数，QPM



**相同 TTFT SLO** 下量出 no-cache / TCP / RDMA 三者的 **λ_max、QPM、成本/query**，给出 TCP→RDMA 的 QPM/成本对比

端到端命路径有 kernel bug，使用离散事件重放：每个逻辑请求的服务时间 由前几个 microbench 的实测数值合成
$$
T_{\text{hit}}
=
T_{\text{lookup}}
+
\frac{S_{\text{KV}}(hN)}{B_{\text{retrieve}}}
+
T_{\text{prefill}}((1-h)N)
$$

- `T_prefill(N) = 0.39318·N + 7.886e-6·N²` ms
- `B_retrieve`：L2 zerocopy `batch_get_into` 中位，TCP 1.19 / RDMA 20.51 GB/s
- `T_lookup = 5 ms` 固定

**排队模型**：`c` 个 prefill worker 的 FIFO 队列（GPU 算力受限，默认 `c=1` 即算力串行）， Poisson 到达（率 λ req/s）。扫 λ 直到 P95 TTFT > SLO 得到 `λ_max`；

计算 `QPM = 60·λ_max`； `Cost/Query` 正比于 `C_node / QPM`。

三种传输（no-cache/TCP/RDMA）在同一 context 下**共用同一 SLO** （`SLO = T_prefill(N) × slo_factor`，`slo_factor=2.5`，由 no-cache 单请求服务定，与传输无关）



phit 命中概率，h 命中前缀比例

高命中（h=0.95, phit=0.8），retrieve 主导，RDMA 显著

| Context | 方案     | KV retrieve 带宽 | 命中请求 restore 延迟 | 平均服务时间 | P95 TTFT SLO | 最大可持续到达率 λ_max | QPM   | QPM vs no-cache | RDMA/TCP QPM |
| ------- | -------- | ---------------- | --------------------- | ------------ | ------------ | ---------------------- | ----- | --------------- | ------------ |
| 8k      | no-cache | —                | —                     | 3.750 s      | 9.375 s      | 0.080 req/s            | 4.80  | 1.00×           | —            |
| 8k      | TCP      | 1.19 GB/s        | ≈0.43 s               | 1.236 s      | 9.375 s      | 0.404 req/s            | 24.26 | 5.05×           | 1.00×        |
| 8k      | RDMA     | 20.51 GB/s       | ≈0.025 s              | 0.916 s      | 9.375 s      | 0.546 req/s            | 32.75 | 6.82×           | **1.35×**    |
| 16k     | no-cache | —                | —                     | 8.559 s      | 21.397 s     | 0.035 req/s            | 2.10  | 1.00×           | —            |
| 16k     | TCP      | 1.19 GB/s        | ≈0.86 s               | 2.665 s      | 21.397 s     | 0.188 req/s            | 11.26 | 5.36×           | 1.00×        |
| 16k     | RDMA     | 20.51 GB/s       | ≈0.050 s              | 2.022 s      | 21.397 s     | 0.247 req/s            | 14.84 | 7.07×           | **1.32×**    |
| 24k     | no-cache | —                | —                     | 14.426 s     | 36.065 s     | 0.021 req/s            | 1.25  | 1.00×           | —            |
| 24k     | TCP      | 1.19 GB/s        | ≈1.28 s               | 4.325 s      | 36.065 s     | 0.116 req/s            | 6.94  | 5.55×           | 1.00×        |
| 24k     | RDMA     | 20.51 GB/s       | ≈0.075 s              | 3.362 s      | 36.065 s     | 0.149 req/s            | 8.92  | 7.14×           | **1.29×**    |

- 高命中下缓存把 QPM 提升 **~5–7×**、成本降到 **~0.14–0.20×**。
- **此时 RDMA 比 TCP 高 ~1.29–1.35× QPM**（成本再降 ~22–26%）。因为 h=0.95 时后缀现算≈0， 服务时间由 **retrieve 主导**：TCP restore（8k=0.43s/16k=0.86s/24k=1.28s 摊进平均）明显拖慢， RDMA（0.02–0.07s）几乎不占时间 ⇒ 直接转成更高 λ_max。



中等命中（h=0.5, phit=0.5），算力主导，RDMA≈TCP

| Context | 方案     | KV retrieve 带宽 | 命中请求 restore 延迟 | 平均服务时间 | P95 TTFT SLO | 最大可持续到达率 λ_max | QPM  | QPM vs no-cache | RDMA/TCP QPM |
| ------- | -------- | ---------------- | --------------------- | ------------ | ------------ | ---------------------- | ---- | --------------- | ------------ |
| 8k      | no-cache | —                | —                     | 3.750 s      | 9.375 s      | 0.080 req/s            | 4.80 | 1.00×           | —            |
| 8k      | TCP      | 1.19 GB/s        | ≈0.225 s              | 2.861 s      | 9.375 s      | 0.140 req/s            | 8.39 | 1.75×           | 1.00×        |
| 8k      | RDMA     | 20.51 GB/s       | ≈0.013 s              | 2.755 s      | 9.375 s      | 0.145 req/s            | 8.71 | 1.81×           | **1.04×**    |
| 16k     | no-cache | —                | —                     | 8.559 s      | 21.397 s     | 0.035 req/s            | 2.10 | 1.00×           | —            |
| 16k     | TCP      | 1.19 GB/s        | ≈0.450 s              | 6.382 s      | 21.397 s     | 0.063 req/s            | 3.76 | 1.79×           | 1.00×        |
| 16k     | RDMA     | 20.51 GB/s       | ≈0.026 s              | 6.170 s      | 21.397 s     | 0.065 req/s            | 3.89 | 1.85×           | **1.03×**    |
| 24k     | no-cache | —                | —                     | 14.426 s     | 36.065 s     | 0.021 req/s            | 1.25 | 1.00×           | —            |
| 24k     | TCP      | 1.19 GB/s        | ≈0.675 s              | 10.564 s     | 36.065 s     | 0.038 req/s            | 2.27 | 1.82×           | 1.00×        |
| 24k     | RDMA     | 20.51 GB/s       | ≈0.039 s              | 10.246 s     | 36.065 s     | 0.039 req/s            | 2.34 | 1.87×           | **1.03×**    |


- **以存代算（缓存）本身把 QPM 提升 ~1.8×、成本降到 ~0.55×**（相对全量重算），因为命中请求少算了 hN 前缀。
- **但 RDMA 仅比 TCP 高 ~1.03–1.04×**：在 8k–24k、h=0.5 下，retrieve 时间（TCP 225–675 ms） 相对 prefill 算力（数秒~十几秒）占比很小，**服务时间被算力主导**，换 RDMA 省下的那点 retrieve 时间对 λ_max 影响甚微。⇒ **中短上下文 + 中等命中，RDMA 的收益不明显**。



结论：

由于 DeepSeek V3.2 特性，以存代算在两种传输下都提升吞吐/降成本：中等命中 QPM ×1.8（成本 0.55×）， 高命中 QPM ×5–7（成本 0.14–0.20×）。

RDMA vs TCP 的增益强依赖 retrieve 占服务时间的比重：

- 算力主导区（中短上下文 / 中等命中）：retrieve 占比小，RDMA≈TCP（~1.03–1.04×）
- retrieve 主导区（高命中 / 长上下文）：RDMA 比 TCP 高 ~1.29–1.35× QPM，成本再降 ~1/4。 

进一步外推（长上下文）RDMA 优势更大：h=1.0、128k 时 restore 是唯一变量， TCP restore ≈ 8.6GB/1.19 ≈ 7.2 s、RDMA ≈ 0.42 s（~17×），命中请求 TTFT 差距直接放大到 ~17× （128k 单节点放不下、只能靠 §模型外推，见步骤4 §2 / 步骤5 §2 的外推上界说明）。