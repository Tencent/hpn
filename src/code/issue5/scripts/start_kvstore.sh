#!/bin/bash
# =============================================================================
# Issue 5 - 远端 KV 存储节点(Mooncake L3 Store)启动脚本
# -----------------------------------------------------------------------------
# 只起 mooncake_master + mooncake_store_service, 不跑模型
# 用法:
#   MC_PROTOCOL=rdma MC_SEGMENT_SIZE=200gb bash start_kvstore.sh
#
# 可选:
#   MC_PROTOCOL      rdma(默认) | tcp   —— 需与 prefill 端一致
#   MC_SEGMENT_SIZE  贡献给全局 KV 池的本机内存, 默认 200gb, 按可用内存调整
#
# 启动: kubectl get pod -o wide 拿到本 pod 宿主机 IP,
# 填到 prefill 端 start_prefill.sh 的 MOONCAKE_STORAGE_NODE_IP。
# =============================================================================
set -euo pipefail

set +u
. ~/.bashrc || true
set -u

export MC_CONDA_ENV="${MC_CONDA_ENV:-python310_torch29_cuda}"
export PATH="/root/miniconda/envs/${MC_CONDA_ENV}/bin:$PATH"

export HOST_IP=$(hostname -i)
export MC_PROTOCOL="${MC_PROTOCOL:-rdma}"
export MC_SEGMENT_SIZE="${MC_SEGMENT_SIZE:-200gb}"
# metadata / store 端口(默认 18080/18081; 8080/8081 常被同宿主机其它 hostNetwork pod 占用)
# 注意: 改这里后 prefill 端 start_prefill.sh 的 MC_META_PORT 需保持一致
export MC_META_PORT="${MC_META_PORT:-18080}"
export MC_STORE_PORT="${MC_STORE_PORT:-18081}"

# mooncake rdma 网卡自动发现; 如需指定, 设 MC_MS_AUTO_DISC=0 并给 MOONCAKE_DEVICE 传网卡名
export MC_MS_AUTO_DISC="${MC_MS_AUTO_DISC:-1}"

echo "[issue5-kvstore] HOST_IP=${HOST_IP} protocol=${MC_PROTOCOL} segment=${MC_SEGMENT_SIZE} meta_port=${MC_META_PORT} store_port=${MC_STORE_PORT}"

python3 -c "import mooncake; print('[issue5-kvstore] mooncake OK:', mooncake.__file__)"

# ---- 1) 启动 master(内嵌 http metadata server: ${MC_META_PORT}; master RPC: 50051)----
mooncake_master \
    --enable_http_metadata_server=true \
    --http_metadata_server_port=${MC_META_PORT} \
    --eviction_high_watermark_ratio=0.95 \
    > /tmp/mooncake_master.log 2>&1 &
MASTER_PID=$!
echo "[issue5-kvstore] mooncake_master started, pid=${MASTER_PID}, log=/tmp/mooncake_master.log"
sleep 8

# ---- 2) 启动 store service, 贡献本机内存到全局 KV 池 ----
# 本镜像里的 mooncake 0.3.5 store_service 不再读 MOONCAKE_* 环境变量,
# 而是从 --config <json> (或 MOONCAKE_CONFIG_PATH) 读配置, 可用 -D key=value 覆盖。
# 纯存储节点不发起 get/put, 故 local_buffer_size=0。
# global_segment_size 需为字节整数, 这里把 MC_SEGMENT_SIZE(如 200gb) 换算成字节。
seg_bytes() {
  local s="${1,,}"; local n="${s//[a-z]/}"
  case "$s" in
    *tb) echo $(( n * 1024 * 1024 * 1024 * 1024 ));;
    *gb) echo $(( n * 1024 * 1024 * 1024 ));;
    *mb) echo $(( n * 1024 * 1024 ));;
    *)   echo "$n";;
  esac
}
MC_SEGMENT_BYTES=$(seg_bytes "$MC_SEGMENT_SIZE")

MC_CFG=/tmp/mooncake_kvstore.json
cat > "$MC_CFG" <<EOF
{
  "local_hostname": "${HOST_IP}",
  "metadata_server": "http://${HOST_IP}:${MC_META_PORT}/metadata",
  "global_segment_size": ${MC_SEGMENT_BYTES},
  "local_buffer_size": 0,
  "protocol": "${MC_PROTOCOL}",
  "device_name": "",
  "master_server_address": "${HOST_IP}:50051"
}
EOF
echo "[issue5-kvstore] store config -> $MC_CFG (segment=${MC_SEGMENT_BYTES} bytes):"
cat "$MC_CFG"

python3 -m mooncake.mooncake_store_service --config "$MC_CFG" --port=${MC_STORE_PORT}
