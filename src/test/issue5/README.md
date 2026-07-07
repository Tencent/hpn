# 推理 Prefill KV Cache "以存代算"带宽临界值建模与 RDMA 加速

## 如何参与ISSUE

1、如果你愿意解决issue，请在腾讯开源研学基地「领取issue任务」

2、Fork 到个人的仓库下

3、在个人仓库解决完对应的任务后，将对应代码提交到 /src/code/issue5/ 目录

4、PR提交后，项目导师将进行 code review， PR 被合并后即视为任务完成

5、如有任何疑问可以在评论区留言或者邮件至联络人

## Issue 目标

1. 对推理 prefill 阶段"从远端存储加载 KV Cache 替代重算"的方案建立带宽临界值模型
2. 将 KV Cache 传输由 TCP 替换为 RDMA，量化其对 TTFT 和 QPM/成本的影响

## 技术背景

在 prefill 阶段，若能从共享远端存储加载已命中的 KV Cache，可省去重算耗时。但存在带宽临界值：当「命中加载耗时 + 未命中部分重算耗时」大于「全量重算耗时」时，以存代算失去优势，该平衡点对应的带宽即临界带宽（与命中率、KV Cache 大小、算力相关）。线上实测（如 128K 分桶、命中率约 30%）显示当前 TCP 访问带宽偏低（约 2.5 GB/s），framework 侧认为若 TCP→RDMA 提升带宽、降低延迟，可在给定 TTFT 内加大并发 query 数，提升 QPM、降低成本。

## 环境准备

1. 你可以参考开源 KV Cache 复用/存储框架（如 LMCache / vLLM 的 KV connector 机制）了解 lookup / retrieve / store 流程
2. 你需要在两台 GPU 服务器上搭建"prefill 节点 + 共享远端 KV 存储"的测试环境，跑通基于 TCP 的 KV Cache 加载基线
3. 你需要采集不同命中率、不同上下文长度（如 8K/32K/128K）下的加载耗时、重算耗时数据

## 验收要求

1. 提交带宽临界值模型（给定命中率、上下文长度、KV Cache 量、算力，求出使以存代算划算的临界带宽），并用实测数据验证
2. 提交将 KV Cache 传输从 TCP 改为 RDMA 的实现，并给出相同 TTFT 约束下带宽、延迟、可承载并发 query 数与 QPM 的对比报告
