# 集合通信 Bitwise 可复现性诊断工具

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue4/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 构建一套针对集合通信（AllReduce / Reduce-Scatter）的 bitwise 可复现性诊断工具，能在同配置 run-to-run 对照下自动定位首次出现位级差异的通信调用

## 技术背景

大规模训练中常见问题不是"跑不快"而是"偶发不一致难复现"。实践中观察到：两次完全同配置（相同初始权重、相同数据流、相同超参）的训练，前若干 step loss 逐位一致，但从中段某 step 起开始出现 1e-5→1e-3 量级的微小差异并累积放大，且无 NaN/skip。此类差异最可能来自集合通信非确定性的浮点归约顺序（如 ring/tree/PAT 算法或拓扑切换导致的累加顺序变化）。需要工具化手段快速定位到底是哪一次通信调用引入了不确定性。

## 环境准备

1. 你可以下载 NCCL 源代码：https://github.com/NVIDIA/nccl 与 nccl-tests：https://github.com/NVIDIA/nccl-tests
2. 你需要了解影响确定性的关键因素：`NCCL_ALGO`/`NCCL_PROTO`、拓扑固定、`torch.use_deterministic_algorithms`、`CUBLAS_WORKSPACE_CONFIG` 等
3. 你需要设计一个可对同一份输入张量重复执行 AllReduce 并逐位比对输出的测试框架，支持注入不同算法/协议/拓扑

## 验收要求

1. 提供一个诊断工具/脚本：对同配置的两次（或多次）集合通信运行，逐位比对结果，输出"首次出现位级差异的调用点、差异量级、差异随规模/迭代的演化"
2. 给出至少一个可复现的 non-determinism 案例，并利用该工具定位到引入不确定性的具体算法/协议配置，同时给出一组能达到 run-to-run bitwise 一致的通信配置建议
