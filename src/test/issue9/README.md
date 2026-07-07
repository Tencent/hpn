# 推荐场景多表 AlltoAllv 聚合融合算子

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue9/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 在推荐/广告训练场景下，将多张 Embedding 表各自独立发起的 AlltoAllv 通信聚合为一次融合通信，降低 kernel launch 次数与控制面开销，提升整体 A2A 吞吐

## 技术背景

推荐/广告训练中，稀疏 Embedding 通常由多张表构成，每张表的特征交换会各自发起一次 AlltoAllv。表数量多时，逐表通信带来大量 kernel launch、控制面协商与小消息，难以打满带宽。将多表的 AlltoAllv 在数据面聚合为一次通信（统一 split/displacement 计算、合并 send/recv buffer），可显著减少启动与协商开销。这是 A2A 通信库演进中"多表 AlltoAllv 聚合"的目标方向之一。

## 环境准备

1. 你可以参考开源 all-to-all 通信实现（如 DeepEP 的 alltoall、Horovod 的 `hvd.alltoall` 接口语义）了解 split_sizes / displacements 的组织方式
2. 你需要在多机多卡环境构造"多张 Embedding 表、各表 first-dim 大小不同"的 AlltoAllv 工作负载
3. 你需要跑通"逐表分别 AlltoAllv"的基线，采集其通信时间与 launch 次数

## 验收要求

1. 实现一个多表 AlltoAllv 聚合融合算子：一次调用完成多张表的特征交换，正确性与逐表方案一致
2. 在相同工作负载下，给出聚合方案相对逐表基线在 kernel launch 次数、控制面开销、端到端通信时间上的改善数据（要求端到端通信时间明显下降）
