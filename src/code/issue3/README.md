# Issue 3：AlltoAllv Send/Recv 分阶段 Overlap

## 编译运行

```bash
cd src/code/issue3
make                                    # 编译（nvcc，需GPU）
./phased_alltoall_sim --mode compare    # 分阶段 vs 基线对比
./phased_alltoall_sim --mode sweep      # 扫描不同recv SM数
make test                               # 单元测试
cd ../../test/issue3 && python overlap_test.py   # Python脚本
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num-ranks` | 8 | 总rank数 |
| `--num-tokens` | 128 | 每rank token数 |
| `--hidden` | 7168 | 隐藏层维度 |
| `--send-sms` | 24 | 发送阶段SM数 |
| `--recv-sms` | 4 | 接收阶段SM数 |
| `--slow-ranks` | 2 | 慢卡数量 |
| `--slow-delay-ms` | 2.0 | 慢卡延迟(ms) |

## 结果

```
8rank/2慢卡(2ms) → 端到端提升28.87%
recv=4 SM最优，释放20 SM给GEMM
```

## 参考

- DeepEP `csrc/kernels/legacy/internode_ll.cu` — phases 位掩码内核
- DeepEP `csrc/legacy/buffer.hpp` — `return_recv_hook` 机制
