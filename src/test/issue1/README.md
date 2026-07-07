# 复现并优化 DeepEP V2 的 Analytical SM/QP 预算模型

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue1/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 在维持通信带宽的前提下，用"分析式模型"推导 EP dispatch/combine 通信所需的 SM 数与 QP 数，替代人工扫参，为 GEMM 计算腾出更多 SM 资源

## 技术背景

MoE 的 Expert Parallel 通信 kernel 若占用过多 SM，会压缩同时运行的 GEMM 可用资源，拉低端到端 MFU。DeepEP V2 用 `get_theoretical_num_sms()` 以分析式方式推导 SM 预算：按通信阶段显式累加 HBM 读写、RDMA、NVLink 流量，比较 `rdma_traffic/rdma_gbs` 与 `nvlink_traffic/nvlink_gbs` 识别瓶颈链路并反推所需 SM 数，再乘约 1.25 安全系数、偶数对齐、设置下限（如 4 SM，overlap 关闭时抬到 64 SM）。QP 预算同理：direct 取 `min(num_sms, 8+1)`，hybrid 取 `num_sms*16+1`。

## 环境准备

1. 你可以在 GitHub 下载 DeepEP 开源代码：https://github.com/deepseek-ai/DeepEP
2. 你需要在两台（或多台）机器上部署 DeepEP，跑通 `test_internode.py`，达到性能基线
3. 你需要阅读并理解 `get_theoretical_num_sms()` 的流量累加与瓶颈识别逻辑
4. 你可以固定一组 EP 配置（专家数、top-k、节点数、隐藏维），扫描不同 SM 数，记录带宽随 SM 变化曲线

## 验收要求

1. 给出该分析式 SM/QP 预算模型的推导说明文档，以及在给定 EP 配置下模型预测值与实测最优 SM 数的对比曲线
2. 在带宽不低于人工扫参最优值 95% 的前提下，用模型自动给出的 SM 数完成 dispatch/combine，并给出相比 24 SM 固定配置节省的 SM 数与端到端收益分析
