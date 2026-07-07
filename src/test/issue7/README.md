# EP 通信库 Scale-Up / Scale-Out 两层 Team 抽象与层次化路由实现

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue7/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 为 EP all-to-all 通信实现"scale-out（节点间/Rail 域）+ scale-up（节点内/NVLink 域）"两层 team 抽象，使上层业务无需感知多 rail/多 plane 网络细节
2. 基于该抽象实现 hybrid dispatch 的两级计数规约与二级转发，验证在多节点下相比扁平 direct 模式的可扩展性

## 技术背景

在大规模 MoE 集群中，把"网络多 rail / 多 plane"复杂度直接暴露给业务是不可持续的。DeepEP V2 用 `ncclTeamTagRail`（scale-out）与 `ncclTeamTagLsa`（scale-up）把拓扑建模进 backend team 抽象，kernel 写成"scale-out 子团 + scale-up 子团"的泛化逻辑：notify warp 做两级计数规约（rail 内聚合 + LSA 内聚合），scaleout warp 负责跨节点发送，forward warp 负责节点内二级转发并构建回程元数据。

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP，重点阅读 hybrid dispatch/combine 相关 kernel
2. 你需要在多节点（≥2 节点，节点内多卡）环境部署并跑通 internode 测试
3. 你需要理解 hybrid 模式中 notify/scaleout/forward 三类 warp 的职责与 `token_metadata_at_forward`、`channel_linked_list` 等元数据结构

## 验收要求

1. 给出 scale-up / scale-out 两层 team 抽象的接口设计与 hybrid dispatch 两级计数规约 + 二级转发的实现
2. 在多节点环境下，验证 hybrid 模式相比扁平 direct 模式在扩大节点数时的带宽/可扩展性优势，并给出 combine 反向 replay 结果正确性的验证
