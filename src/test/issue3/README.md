# AlltoAllv 的 send/recv 分阶段 Overlap 优化（快慢卡场景）

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue3/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 在计算通信 overlap 且存在明显"快慢卡"现象的 AlltoAllv 场景下，通过把通信拆分为发送阶段与接收阶段并分别配置 SM 资源，降低通信对计算 kernel 的抢占，提升端到端 overlap 效率

## 技术背景

在推荐/MoE 的 AlltoAllv 通信中，各 rank 到达时间不一致（快慢卡），若接收阶段长时间等待远端数据且占用大量 SM，会持续抢占与之 overlap 的计算 kernel 资源。将 AlltoAllv 拆成"发送阶段（拷贝到 rdma send buffer 并提交网络请求）"与"接收阶段（等待远端到达并拷贝到 output tensor）"，并让接收阶段使用较少 SM，可减少这种抢占。DeepEP low-latency 及 UCL-AlltoAllv 扩展接口均支持 send/recv 分离与 per-phase 资源配置（如 recv_phase 默认 4 SM）。

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP，参考 `test_low_latency.py`
2. 你需要在两台机器上部署并跑通 low-latency alltoall 测试，达到性能基线
3. 你需要构造一个"计算 kernel 与 alltoallv 通信 overlap 且人为引入快慢卡（部分 rank 延迟到达）"的测试用例
4. 你需要实现 send/recv 分阶段执行，并对接收阶段的 SM 数（kMaxNumSMs）进行可配置化

## 验收要求

1. 提供可复现快慢卡场景的 overlap 测试脚本，以及 send/recv 分阶段执行的实现
2. 在同样的快慢卡负载下，相比"单 kernel 完成 send+recv 且接收阶段占满 SM"的基线，分阶段并降低接收阶段 SM 的方案在端到端时间（含被 overlap 的计算）上有可量化改善，且通信正确性不变
