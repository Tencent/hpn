# 机内 AllReduce 中 Multimem(NVLS) 与纯 P2P 路径的最优切换阈值探究

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue2/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 针对机内（NVLink/NVSwitch 域）AllReduce，量化 One-Shot、Two-Shot(Multimem) 与纯 P2P 三类实现在不同 message size 下的性能与 SM 占用
2. 建立按消息大小自动选择最优算法的切换阈值模型

## 技术背景

机内 AllReduce 存在多种实现：One-Shot（AllGather + 本地 Reduce，每 GPU 收 (N-1)·M 数据，延迟 O(1)，适合极小消息）；Two-Shot（Reduce-Scatter + AllGather，每 GPU 收发 ~2(N-1)/N·M）。其中 Multimem 版借助 NVSwitch 在交换芯片上完成规约，SM 几乎空闲；纯 P2P 版规约在 GPU SM 上完成，SM 占用更高。不同消息区间下最优算法不同，需要一个明确的切换阈值。

## 环境准备

1. 你可以下载 NCCL 源代码：https://github.com/NVIDIA/nccl 与 nccl-tests：https://github.com/NVIDIA/nccl-tests
2. 你可以了解 NCCL NVLS（NVLink SHARP / multimem）相关环境变量（如 `NCCL_NVLS_ENABLE`）
3. 你需要在单机多卡（含 NVSwitch）环境上，分别测试 One-Shot / Two-Shot(Multimem) / 纯 P2P 三类实现在 message size 从 1KB 到 1GB 区间的 AllReduce 性能与 SM 占用

## 验收要求

1. 输出三类机内 AllReduce 实现在各 message size 下的通信完成时间与 SM/Warp 占用测试报告
2. 给出按消息大小自动选择算法的切换阈值模型（含决策边界），并在测试环境中验证按该阈值动态选择相比固定使用单一算法的整体收益
