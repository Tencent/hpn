# Issue 1：分析式 SM/QP 预算模型

## 编译运行

```bash
cd src/code/issue1
make                              # 编译（g++，无需GPU）
./sm_budget_model --mode single   # 单次计算
./sm_budget_model --mode sweep    # SM扫描输出CSV
./sm_budget_model --mode compare  # 与固定24SM对比
make test                         # 单元测试
```

## 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--num-experts` | 288 | 总专家数 |
| `--num-topk` | 8 | 每token topk |
| `--num-scaleout-ranks` | 8 | 跨节点rank数 |
| `--num-scaleup-ranks` | 1 | 节点内rank数 |
| `--num-device-sms` | 132 | 设备SM总数 |
| `--rdma-gbs` | 0(自动50) | RDMA带宽 |
| `--nvlink-gbs` | 0(自动450) | NVLink带宽 |

## 结果

```
288专家/topk8/8节点 → 推荐4 SM，节省20 SM vs 固定24 SM，带宽不减
```

## 参考

DeepEP `deep_ep/buffers/elastic.py` — `get_theoretical_num_sms()` (行 729-853)
