#!/bin/bash
set -e

# 容器名称 / 镜像
container_name="docker-env-flux"
image="${IMAGE:-nvcr.io/nvidia/pytorch:24.07-py3}"

# ===== NVSHMEM/UCX 相关可调参数（按需修改/传入环境变量覆盖） =====
# 对称堆大小（NVSHMEM必须）：按需改为 2G/4G/8G 等
NVSHMEM_SIZE="${NVSHMEM_SIZE:-4G}"
# UCX 传输层（单机多卡 + 可回落跨节点，请按需精简）
UCX_TLS_DEFAULT="${UCX_TLS_DEFAULT:-cuda_ipc,sm,self,rc,tcp}"
# UCX 网卡选择（有多网卡时可指定，如 'mlx5_0:1'；不需要就留空）
UCX_NET_DEVICES="${UCX_NET_DEVICES:-}"
# 开启信息输出便于自检
UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-info}"
NVSHMEM_INFO="${NVSHMEM_INFO:-1}"

# 可选：若主机已安装 NVSHMEM/HPC-X，可通过下列路径挂载进容器
NVSHMEM_HOST_DIR="${NVSHMEM_HOST_DIR:-/opt/nvshmem}"
HPCX_HOST_DIR="${HPCX_HOST_DIR:-/opt/hpcx}"

# 根据主机实际情况组装额外挂载/环境
extra_mounts=""
extra_envs=""

if [ -d "$NVSHMEM_HOST_DIR" ]; then
  extra_mounts+=" -v $NVSHMEM_HOST_DIR:/opt/nvshmem"
  # 将 NVSHMEM bin/lib 放进容器 PATH/LD_LIBRARY_PATH（注意：这里在容器内生效）
  extra_envs+=" -e PATH=/opt/nvshmem/bin:\$PATH -e LD_LIBRARY_PATH=/opt/nvshmem/lib:\$LD_LIBRARY_PATH"
fi

if [ -d "$HPCX_HOST_DIR" ]; then
  extra_mounts+=" -v $HPCX_HOST_DIR:/opt/hpcx"
  extra_envs+=" -e PATH=/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin:\$PATH -e LD_LIBRARY_PATH=/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib:\$LD_LIBRARY_PATH"
fi

# ====== 容器存在就复用，不存在就创建 ======
if docker inspect "$container_name" >/dev/null 2>&1; then
    echo "$container_name"
    docker start "$container_name" >/dev/null
    docker exec -it "$container_name" /bin/bash
else
    echo "Container does not exist. Creating $container_name ..."

    # 说明：
    # --ulimit memlock=-1, --cap-add=IPC_LOCK: 允许 RDMA 锁页，NVSHMEM/UCX 常用
    # --ipc=host, --shm-size: 共享内存（sm/IPC）依赖
    # --device=/dev/infiniband: RDMA 设备
    # --network=host: 单机/多机低延迟网络
    # --privileged=true: 覆盖多类设备权限（如需收紧可改为更细粒度 --cap-add）
    # -v `pwd`:/home/workcode: 将当前目录挂载进入容器

    docker run \
        --name "$container_name" \
        -itd \
        --gpus all \
        --device=/dev/infiniband \
        --network=host \
        --ipc=host \
        --shm-size 500G \
        --security-opt seccomp=unconfined \
        --privileged=true \
        --ulimit memlock=-1 \
        --cap-add=IPC_LOCK \
        -v "$(pwd)":/home/workcode \
        $extra_mounts \
        --workdir /home/workcode \
        -e NVSHMEM_SYMMETRIC_SIZE="$NVSHMEM_SIZE" \
        -e NVSHMEM_INFO="$NVSHMEM_INFO" \
        -e UCX_TLS="$UCX_TLS_DEFAULT" \
        -e UCX_LOG_LEVEL="$UCX_LOG_LEVEL" \
        $( [ -n "$UCX_NET_DEVICES" ] && echo -e "-e UCX_NET_DEVICES=$UCX_NET_DEVICES" ) \
        -e NCCL_NET=UCX \
        $extra_envs \
        "$image" /bin/bash

    echo "Container $container_name created."
    docker exec -it "$container_name" /bin/bash
fi
