# 集合通信算法（Ring/Tree/NVLS）拓扑感知自动选择

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue10/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 针对 AllReduce/AllGather/ReduceScatter，构建一个"消息大小 × 拓扑规模"感知的算法自动选择器，在给定硬件拓扑下自动选出 Ring / Tree / NVLS(multimem) 中最优的算法，逼近各消息区间的性能上限

## 技术背景

不同集合通信算法在不同 message size 与拓扑规模下表现差异巨大：Ring 步骤数 O(N)、带宽利用好但小消息延迟高；Tree 小消息延迟低；NVLS(multimem) 借助 NVSwitch 硬件规约在机内中大消息上 SM 占用极低。NCCL 内部有算法/协议选择表，但默认调优表未必贴合星脉的实际拓扑与业务消息分布。需要一个可离线标定 + 在线选择的机制，以逼近各区间性能上限。

## 环境准备

1. 你可以下载 NCCL 源代码：https://github.com/NVIDIA/nccl 与 nccl-tests：https://github.com/NVIDIA/nccl-tests
2. 你需要了解 `NCCL_ALGO`（Ring/Tree/NVLS/PAT 等）与 `NCCL_PROTO`（Simple/LL/LL128）的作用与切换方式
3. 你需要在目标拓扑（单机多卡含 NVSwitch，可选多机）上，对各算法/协议在 message size 从 1KB 到 1GB 区间做基线性能标定

## 验收要求

1. 输出各集合通信原语在"算法 × 协议 × message size × 拓扑规模"维度上的性能标定数据，以及据此生成的选择表
2. 实现一个自动选择器（离线标定 + 在线按消息大小/规模查表选择），并验证：在混合消息大小的工作负载上，自动选择相比固定使用单一算法（如仅 Ring）在整体通信时间上有可量化提升
