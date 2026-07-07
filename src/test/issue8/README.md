# MoE Combine 本地 Reduce 三种模式的精度/流量权衡分析

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue8/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 系统对比 MoE combine 阶段三种归约模式在网络流量、通信完成时间与数值精度上的差异，给出不同场景下的选型建议

## 技术背景

DeepEP V2 direct combine 存在三类工作模式：
- 模式 A（no-expand / 无需本地 reduce）：一个返回 token 只对应一个来源，直接 load/store 写回，路径最短、精度最好；
- 模式 B（expand + allow_multiple_reduction）：本地先把多个 top-k 副本做 shared memory 向量化 reduction，减少网络返回流量；
- 模式 C（expanded send，不允许本地 reduce）：每个副本都回传，由 epilogue 在最终位置再做 reduction，本质是"用更多网络流量换更少本地归并误差"。

三种模式在流量、时延、浮点归约误差上各有取舍，需要量化对比。

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP，跑通 combine 相关测试
2. 你需要理解 `combine_reduce` / `combine_reduce_epilogue` 以及三种模式的触发条件
3. 你需要构造可控的 top-k 分布（如不同的重复率），以放大三种模式的差异

## 验收要求

1. 提供三种 combine 模式在相同输入下的对比测试脚本，输出各模式的网络返回流量、combine 通信完成时间、以及相对高精度基线（如 FP32 全量回传后归约）的数值误差
2. 给出一张"按 top-k 重复率 / 消息大小 / 精度要求"选择 combine 模式的决策表，并用测试数据支撑其结论
