#!/bin/bash
# =============================================================================
# Issue 5 - 计算节点(Prefill)启动脚本 —— DeepSeek-V3.2 (MLA + NSA/DSA) 版
# -----------------------------------------------------------------------------
# DeepSeek-V3.2(DeepseekV32ForCausalLM, MLA + DSA
# 稀疏索引), 其 KV 池是 NSATokenToKVPool —— HiRadixCache 明确支持
# (mem_cache/hiradix_cache.py: isinstance(..., NSATokenToKVPool) -> NSATokenToKVPoolHost)。
#
# 用法:
#   MOONCAKE_STORAGE_NODE_IP=<存储节点宿主机IP> MC_PROTOCOL=tcp bash start_prefill_v32.sh
# =============================================================================
set -euo pipefail

# ---- 参数校验 ----
if [[ -z "${MOONCAKE_STORAGE_NODE_IP:-}" ]]; then
  echo "[ERROR] 请先设置 MOONCAKE_STORAGE_NODE_IP=<存储节点宿主机IP>" >&2
  exit 1
fi
export MC_PROTOCOL="${MC_PROTOCOL:-tcp}"
export MC_META_PORT="${MC_META_PORT:-18080}"
echo "[issue5-prefill-v32] storage_ip=${MOONCAKE_STORAGE_NODE_IP} protocol=${MC_PROTOCOL} meta_port=${MC_META_PORT}"

# ---- 基础环境 ----
set +u
. ~/.bashrc || true
set -u

# conda env: py3.10 + torch2.9, 预装 mooncake 0.3.5 + sglang(与镜像 glibc2.31 匹配)
export MC_CONDA_ENV="${MC_CONDA_ENV:-python310_torch29_cuda}"
export PATH="/root/miniconda/envs/${MC_CONDA_ENV}/bin:$PATH"

unset XPU_DUMMY_EVENT || true
export PYTHONPATH=/workspace/sglang/python:/workspace/sglang/sgl-kernel/python:${PYTHONPATH:-}
# DeepSeek-V3.2 int8(w8a8) 权重; 与 --quantization w8a8_int8 匹配。

export MODEL_PATH=/root/v32_int8_fix/

export SGLANG_APPLY_CONFIG_BACKUP=none
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MODEL_ID="${MODEL_ID:-p800_deepseek_v32_issue5}"

export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export SGLANG_OPT_USE_TILELANG_MHC_POST=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_IS_FLASHINFER_AVAILABLE=False
export USE_FAST_ALLOC_EXTEND_KUNLUN=False
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=False
export SGLANG_OPT_USE_JIT_NORM=True
export SGLANG_OPT_USE_FUSED_STORE_CACHE=0
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_BF16_FP32_GEMM_ALGO="torch"
export SGLANG_TOPK_TRANSFORM_512_TORCH=False
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_PREP_IN_CUDA_GRAPH=False
export SGLANG_OPT_CP_REARRANGE_TRITON=False
export BKCL_RDMA_VERBS=1
export BKCL_ENABLE_XDR=1
export BKCL_RDMA_NICS="eth1,eth1,eth2,eth2,eth3,eth3,eth4,eth4"
export NCCL_IB_GID_INDEX=3
export XSHMEM_QP_NUM_PER_RANK=32
export XSHMEM_MODE=1
export XSHMEM_SYMMETRIC_SIZE=4294967296
export CUDA_DEVICE_ORDER=OAM_ID
export CUDA_ENABLE_P2P_NO_UVA=1
export OPS_DEBUG_CHECK=1
export SGLANG_TOOL_STRICT_LEVEL=2

export HOST_IP=$(hostname -i)
LOG_DIR=logs_issue5_prefill_v32
mkdir -p "$LOG_DIR"

# ---- 依赖(镜像若已装可注释掉)----
pip3 install "cocopod-1.3.0.torch29-cp310-cp310-linux_x86_64.whl" --force-reinstall

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --tp 8 \
    --ep 8 \
    --port 8000 \
    --quantization w8a8_int8 \
    --trust-remote-code \
    --kv-cache-dtype bfloat16 \
    --dtype bfloat16 \
    --mem-fraction-static 0.83 \
    --chunked-prefill-size 2048 \
    --max-running-requests 4 \
    --max-prefill-tokens 16384 \
    --disable-cuda-graph \
    --skip-server-warmup \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --enable-metrics --collect-tokens-histogram \
    --host "$HOST_IP" \
    --allow-auto-truncate \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --served-model-name "$MODEL_ID" \
    --enable-cache-report \
    --page-size 64 \
    --enable-hierarchical-cache \
    --disable-hicache-numa-detect \
    --hicache-storage-backend mooncake \
    --hicache-mem-layout page_first \
    --hicache-write-policy write_through \
    --hicache-storage-prefetch-policy timeout \
    --hicache-ratio 2 \
    --hicache-storage-backend-extra-config "$HICACHE_MOONCAKE_CFG"
