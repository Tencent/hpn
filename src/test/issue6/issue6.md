# 实现 EP Dispatch 的 Deterministic Slot 分配

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue6/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 为 MoE EP dispatch 实现确定性的 slot 分配：在相同输入下，token 到接收缓冲区 slot 的映射结果完全一致，从而支持 forward/backward 对账、回归复现与线上异常 replay

## 技术背景

DeepEP 原始 dispatch 中，多个 warp/SM 通过 `atomicAdd` 抢占 slot，导致相同输入在不同运行中 slot 分配顺序可能不同，使得偶发不一致难以复现。DeepEP V2 的 `dispatch_deterministic_prologue` 给出了一种确定性协议：先对每个目标 rank/expert 计数 → 做跨 SM 前缀和（prefix sum）→ 再按确定次序回填 slot，把原子竞争替换为确定性分配。该特性不追求带宽，而是为可复现性服务。

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP，跑通 `test_internode.py` / `test_intranode.py`
2. 你需要理解原始 dispatch 中 slot 分配（atomic add 去重）的实现路径
3. 你需要设计一个"固定输入（固定 topk_idx、固定 token）多次运行、比对 slot 分配是否逐一致"的测试

## 验收要求

1. 实现一个 deterministic dispatch 前导逻辑（计数 → 跨 SM 前缀和 → 回填 slot），在相同输入下多次运行 slot 分配结果完全一致
2. 提供一致性验证测试；并给出确定性版本相对原版在 dispatch 带宽上的开销评估（要求带宽下降可控，dispatch 结果正确）
