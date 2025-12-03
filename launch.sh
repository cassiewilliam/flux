#!/bin/bash

# 基础路径配置
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:~/.local/lib/
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
FLUX_SRC_DIR=${SCRIPT_DIR}

# 基础 CUDA/NVSHMEM 配置
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_DISABLE_CUDA_VMM=1
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_MODULE_LOADING=LAZY

# ---- 关键：彻底禁用所有 NCCL 插件 / IB / UCX ----
export NCCL_NET_PLUGIN=none          # 禁用插件
export NCCL_PLUGIN_DISABLE=1         # 禁用HPC-X插件加载
export NCCL_UCX_DISABLE=1            # 禁用 UCX
export NCCL_SHARP_DISABLE=1          # 禁用 SHARP
export NCCL_IB_DISABLE=1             # 禁 IB（单机不需要）
export NCCL_NET=Socket               # 强制走 socket
export NCCL_SHM_DISABLE=0            # 启用共享内存
export NCCL_P2P_DISABLE=0            # 启用PCIe/NVLink P2P
export NCCL_DEBUG=WARN               # 输出最小信息

# UCX安全降级（如果容器内意外带了UCX）
export UCX_TLS=sm,self,cuda_ipc
export UCX_LOG_LEVEL=warn

# Torchrun 设置
nproc_per_node=2
nnodes=1
node_rank=0
master_addr="127.0.0.1"
master_port="23456"
additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

CMD="torchrun \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${FLUX_EXTRA_TORCHRUN_ARGS} ${additional_args} $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret