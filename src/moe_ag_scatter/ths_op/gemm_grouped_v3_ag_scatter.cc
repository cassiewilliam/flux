//===- gemm_grouped_v3_ag_scatter.cc ------------------------------ C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#include "moe_ag_scatter/ths_op/gemm_grouped_v3_ag_scatter.h"
#include "cute/tensor.hpp"
#include "cutlass/util/device_memory.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include "flux/args/moe_ag_scatter.h"
#include "flux/utils.h"
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#include "moe_ag_scatter/sort_util.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <cuda_runtime_api.h>
#include <nvshmemx.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <torch/cuda.h>

namespace bytedance {
namespace flux {
namespace ths_op {

/**
 *
 * 总览：三条流 + 三个事件 + 两类屏障

  流：
  main_stream：当前 PyTorch 的 compute stream（getCurrentCUDAStream()）
  cp_stream_intra_node：节点内 NVSHMEM/拷贝流
  cp_stream_inter_node：跨节点 NVSHMEM/拷贝流

  事件
  ready_event：本地 shard 写入“就绪”
  fetch_remote_event：远端 shard 拉取完成（跨/同节点）
  all_gather_event：整个 all-gather/all-to-all 阶段完成

  屏障/同步
  nvshmemx_barrier_all_on_stream(main_stream)：全局队列式 barrier（含 quiet 语义）
  nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, cp_stream_inter_node)：节点内 barrier（含 quiet）
  barrier_block（device 内存 + CUStreamWriteValue）：给 CUTLASS/GEMM 作为就绪标志数组
 *
 */

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmGroupedV3AGScatterOp::GemmGroupedV3AGScatterOpImpl {
 private:
  const DistEnvTPWithEP tp_env;
  const MoeArguments moe_args;
  const int32_t nexperts_ep;
  const int32_t expert_idx_offset;
  const int32_t ffn_size_shard;

 private:
  torch::Tensor workspace_buffer;
  // we use cutlass::DeviceAllocation instead of pytorch tensor here,
  // because if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set,
  // pytorch tensor .data_ptr() will return a virtual address which is invalid
  // for cuStreamWriteValue32_v2
  cutlass::DeviceAllocation<uint8_t> barrier_block;
  torch::Tensor input_buffer;
  torch::Tensor total_nrows_ep_buffer;
  torch::Tensor sorted_splits;
  std::vector<int> sorted_splits_cpu;
  torch::Tensor gather_index;
  torch::Tensor sorted_scatter_index;
  torch::Tensor sorted_gather_index;

  c10::cuda::CUDAStream cp_stream_intra_node;
  c10::cuda::CUDAStream cp_stream_inter_node;
  cudaEvent_t ready_event;
  cudaEvent_t fetch_remote_event;
  cudaEvent_t all_gather_event;

  void
  init_buffers() {
    this->input_buffer =
        nvshmem_create_tensor({moe_args.max_ntokens, moe_args.hidden}, moe_args.input_dtype);
    auto options = torch::TensorOptions().device(torch::Device(torch::kCUDA));
    this->total_nrows_ep_buffer = torch::empty({1}, options.dtype(c10::ScalarType::Int));
    this->sorted_splits =
        torch::empty({nexperts_ep * tp_env.world_size}, options.dtype(c10::ScalarType::Int));
    this->sorted_splits_cpu.resize(nexperts_ep * tp_env.world_size);
    this->gather_index =
        torch::empty({moe_args.max_ntokens * moe_args.topk}, options.dtype(c10::ScalarType::Int));
    this->sorted_scatter_index =
        torch::empty({moe_args.max_ntokens * moe_args.topk}, options.dtype(c10::ScalarType::Int));
    this->sorted_gather_index =
        torch::empty({moe_args.max_ntokens * moe_args.topk}, options.dtype(c10::ScalarType::Int));
  }

  c10::cuda::CUDAStream
  create_cp_stream() const {
    at::cuda::CUDAGuard guard(at::cuda::current_device());
    cudaStream_t cp_stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&cp_stream, CU_STREAM_NON_BLOCKING));
    return at::cuda::getStreamFromExternal(cp_stream, at::cuda::current_device());
  }

 public:
  GemmGroupedV3AGScatterOpImpl(DistEnvTPWithEP tp_env, MoeArguments moe_args)
      : tp_env(std::move(tp_env)),
        moe_args(std::move(moe_args)),
        nexperts_ep(moe_args.nexperts / tp_env.ep_size),
        expert_idx_offset(nexperts_ep * tp_env.ep_rank),
        ffn_size_shard(moe_args.ffn_hidden / tp_env.ffn_tp_size),
        cp_stream_intra_node(create_cp_stream()),
        cp_stream_inter_node(create_cp_stream()) {
    CHECK_DIV(moe_args.nexperts, tp_env.ep_size);
    CHECK_DIV(moe_args.ffn_hidden, tp_env.ffn_tp_size);
    init_buffers();
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->fetch_remote_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&this->all_gather_event, cudaEventDisableTiming));
  }

  ~GemmGroupedV3AGScatterOpImpl() {
    CUDA_CHECK(cudaEventDestroy(this->all_gather_event));
    CUDA_CHECK(cudaEventDestroy(this->fetch_remote_event));
    CUDA_CHECK(cudaEventDestroy(this->ready_event));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_intra_node));
    CUDA_CHECK(cudaStreamDestroy(this->cp_stream_inter_node));
  }

 protected:
  auto
  get_gemm_meta(bool fast_accum) const {
    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto gemm_layout = _RCR{};
    auto input_dtype = from_torch_dtype(moe_args.input_dtype);
    auto output_dtype = from_torch_dtype(moe_args.output_dtype);
    auto dt_conf = make_gemm_dtype_config(input_dtype, input_dtype, output_dtype, output_dtype);
    auto v3_meta = make_gemm_v3_meta(fast_accum and dt_conf.is_input_fp8());
    auto meta = make_gemm_meta(
        dt_conf, arch, sm_core, _AGScatter{}, gemm_layout, _GemmGroupedV3{}, v3_meta);
    return meta;
  }

  auto
  get_rt_conf() const {
    return make_runtime_config(
        moe_args.max_ntokens * moe_args.topk / moe_args.nexperts, ffn_size_shard, moe_args.hidden);
  }

  /**
   * 执行分布式 all-gather 和 all-to-all 通信操作
   *
   * 该函数实现了跨节点的数据收集和分发，主要功能包括：
   * 1. 从所有节点收集输入数据分片到本地缓冲区
   * 2. 使用 NVSHMEM 进行高效的跨节点通信
   * 3. 通过 CUDA 事件和流实现异步通信同步
   *
   * @param inputs_shard 当前节点的输入数据分片 [ntokens_shard, hidden]
   */
  void
  all_gather_all2all(torch::Tensor const &inputs_shard) {
    using namespace cute;

    // 获取当前节点的 token 数量
    int ntokens_shard = inputs_shard.size(0);

    // 创建完整的输入张量视图，用于存储所有节点的数据
    // 形状: [element_size, hidden, ntokens_shard, world_size]
    Tensor full_input = make_tensor(
        static_cast<uint8_t *>(input_buffer.data_ptr()),
        make_shape(
            make_shape(c10::elementSize(moe_args.input_dtype), moe_args.hidden),
            ntokens_shard,
            tp_env.world_size));

    // 外层循环：按节点索引遍历，优先处理当前节点，然后处理其他节点
    // 内层循环：按本地 rank 遍历，使用 all2all 进行通信
    for (int node_idx = tp_env.node_idx, i = 0; i < tp_env.nnodes;
         ++i, node_idx = (node_idx + 1) % tp_env.nnodes) {
      if (node_idx == tp_env.node_idx) {
        // 处理当前节点：将本地数据复制到共享缓冲区
        auto main_stream = c10::cuda::getCurrentCUDAStream();
        auto shard_input = full_input(_, _, tp_env.rank);

        // 异步复制本地数据到共享缓冲区
        CUDA_CHECK(cudaMemcpyAsync(
            shard_input.data(),
            inputs_shard.data_ptr(),
            shard_input.size(),
            cudaMemcpyDeviceToDevice,
            main_stream));

        // 同步所有节点，确保数据复制完成
        nvshmemx_barrier_all_on_stream(main_stream);

        // 记录数据就绪事件，供其他节点等待
        CUDA_CHECK(cudaEventRecord(this->ready_event, main_stream));
        CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_intra_node, this->ready_event));

      } else {
        // 处理远程节点：从其他节点获取数据
        FLUX_CHECK(nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE) == tp_env.local_rank);

        if (i == 1) {
          // 第一次远程获取时，等待数据就绪事件
          CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_inter_node, this->ready_event));
        }

        // 计算源节点的全局 rank
        int src_rank = tp_env.local_rank_to_global_rank(tp_env.local_rank, node_idx);
        auto shard_input = full_input(_, _, src_rank);

        // 使用 NVSHMEM 从远程节点获取数据
        nvshmemx_getmem_on_stream(
            shard_input.data(),
            shard_input.data(),
            shard_input.size(),
            src_rank,
            this->cp_stream_inter_node);

        // 在节点内同步，确保数据获取完成
        nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, this->cp_stream_inter_node);

        // 记录远程获取完成事件
        CUDA_CHECK(cudaEventRecord(this->fetch_remote_event, this->cp_stream_inter_node));
        CUDA_CHECK(cudaStreamWaitEvent(this->cp_stream_intra_node, this->fetch_remote_event));
      }

      // 内层循环：在节点内进行 all2all 通信
      for (int local_rank = tp_env.local_rank, j = 0; j < tp_env.local_world_size;
           ++j, local_rank = (local_rank + 1) % tp_env.local_world_size) {
        // 计算源节点和目标节点的全局 rank
        int src_rank = tp_env.local_rank_to_global_rank(local_rank, node_idx);
        int local_rank_global = tp_env.local_rank_to_global_rank(local_rank);

        if (local_rank != tp_env.local_rank) {
          // 从其他本地 rank 获取数据（排除自己）
          auto shard_input = full_input(_, _, src_rank);
          nvshmemx_getmem_on_stream(
              shard_input.data(),
              shard_input.data(),
              shard_input.size(),
              local_rank_global,
              this->cp_stream_intra_node);
        }

        // 写入屏障标志，表示该 rank 的数据已就绪
        CU_CHECK(CUStreamWriteValue(
            this->cp_stream_intra_node,
            (CUdeviceptr)(ptr_offset(barrier_block.get(), src_rank * sizeof(int))),
            1,
            CU_STREAM_WRITE_VALUE_DEFAULT));
      }
    }

    // 记录 all-gather 完成事件
    CUDA_CHECK(cudaEventRecord(this->all_gather_event, this->cp_stream_intra_node));
  }

  /**
   * 前向传播的核心实现函数
   *
   * NVSHMEM 与 Kernel 交互机制：
   * 1. 共享内存分配：NVSHMEM 为所有节点分配共享内存空间 (input_buffer)
   * 2. 数据收集：all_gather_all2all() 将所有节点的输入数据收集到共享缓冲区
   * 3. Kernel 访问：GEMM kernel 通过指针直接访问共享缓冲区中的数据
   * 4. 同步机制：使用 CUDA 事件和 NVSHMEM barrier 确保数据一致性
   * 5. 内存管理：共享内存由 NVSHMEM 管理，避免传统 MPI 的通信开销
   *
   * 关键交互点：
   * - ptr_A 指向 input_buffer.data_ptr() (NVSHMEM 共享内存)
   * - Kernel 通过 gather 索引从共享内存读取数据
   * - 通过 barrier_ptr 实现节点间同步
   * - 使用 CUDA 流和事件实现异步通信与计算
   */
  std::vector<torch::Tensor>
  forward_impl(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    // 输入验证：检查输入张量的数据类型、维度和形状
    CHECK_INPUT(inputs_shard, moe_args.input_dtype);
    CHECK_NDIM(inputs_shard, 2);
    int ntokens_shard = inputs_shard.size(0);
    CHECK_2D(inputs_shard, ntokens_shard, moe_args.hidden);

    // 计算全局 token 数量（所有节点的 token 总数）
    int ntokens = ntokens_shard * tp_env.world_size;

    // 验证权重张量
    std::size_t num_weights_group = weights.size();
    for (std::size_t i = 0; i < num_weights_group; ++i) {
      CHECK_INPUT(weights[i], moe_args.input_dtype);
      CHECK_3D(weights[i], nexperts_ep, ffn_size_shard, moe_args.hidden);
    }

    // 验证专家分配信息
    CHECK_INPUT(splits_gpu, torch::kInt32);
    CHECK_NDIM(splits_gpu, 1);
    FLUX_CHECK_GE(splits_gpu.size(0), moe_args.nexperts);

    // 验证散射索引
    CHECK_INPUT(scatter_index, torch::kInt32);
    CHECK_2D(scatter_index, ntokens, moe_args.topk);

    // 验证输出缩放因子（如果提供）
    if (output_scale.has_value()) {
      TORCH_CHECK_EQ(output_scale->size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(output_scale->at(i), torch::kFloat32);
        CHECK_1D(output_scale->at(i), nexperts_ep);
      }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    // 步骤 1: 获取 GEMM 操作符并准备操作缓冲区
    auto meta = this->get_gemm_meta(fast_accum);
    auto rt_conf = this->get_rt_conf();
    OpRegistry::OpPtr op;
    if (hparams.has_value()) {
      op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    {
      // 初始化屏障缓冲区 - 用于节点间同步
      auto tmp_args = GemmGroupedAgScatterArguments{};
      tmp_args.dist_env = tp_env;
      int64_t barrier_size = op->get_barrier_workspace_size(tmp_args);
      barrier_size = (barrier_size + 127) / 128 * 128;
      if (barrier_size > (int64_t)barrier_block.size()) {
        barrier_block.reset(barrier_size);
      }
      CUDA_CHECK(cudaMemsetAsync(barrier_block.get(), 0, barrier_block.bytes()));
    }

    // 步骤 2: 启动 All-gather 通信 - 通过 NVSHMEM 收集所有节点的输入数据
    this->all_gather_all2all(inputs_shard);

    // 步骤 3: 运行辅助 kernel 进行数据预处理
    std::vector<torch::Tensor> outputs;
    {
      int32_t total_nrows_ep;  // 将由 CUDA kernel 初始化
      int32_t *total_nrows_ep_gpu = static_cast<int32_t *>(this->total_nrows_ep_buffer.data_ptr());

      // 计算 gather 索引：确定每个 token 应该从哪个位置读取数据
      calc_gather_index_impl(
          moe_args.nexperts,
          ntokens,
          moe_args.topk,
          expert_idx_offset,
          expert_idx_offset + nexperts_ep,
          static_cast<int32_t const *>(splits_gpu.data_ptr()),
          static_cast<int32_t const *>(scatter_index.data_ptr()),
          static_cast<int32_t *>(gather_index.data_ptr()),
          total_nrows_ep_gpu,
          stream);

      // AG-Scatter 排序：对 token 进行排序以优化内存访问模式
      ag_scatter_sort_impl(
          AGScatterSortOpArguments{
              static_cast<DistEnv>(tp_env),
              ntokens,
              nexperts_ep,
              static_cast<int32_t const *>(splits_gpu.data_ptr()) + expert_idx_offset,
              static_cast<int32_t const *>(gather_index.data_ptr()),
              static_cast<int32_t *>(sorted_splits.data_ptr()),
              static_cast<int32_t *>(sorted_scatter_index.data_ptr()),
              static_cast<int32_t *>(sorted_gather_index.data_ptr()),
          },
          stream);

      CUDA_CHECK(cudaMemcpyAsync(
          &total_nrows_ep, total_nrows_ep_gpu, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaMemcpyAsync(
          sorted_splits_cpu.data(),
          sorted_splits.data_ptr(),
          sizeof(int32_t) * nexperts_ep * tp_env.world_size,
          cudaMemcpyDeviceToHost,
          stream));
      c10::cuda::stream_synchronize(stream);
      // shapes check
      if (outputs_buf.has_value()) {
        outputs = outputs_buf.value();
      } else {
        for (std::size_t i = 0; i < num_weights_group; ++i) {
          outputs.emplace_back(
              torch::empty({total_nrows_ep, ffn_size_shard}, inputs_shard.options()));
        }
      }

      TORCH_CHECK_EQ(outputs.size(), num_weights_group);
      for (std::size_t i = 0; i < num_weights_group; ++i) {
        CHECK_INPUT(outputs[i], moe_args.output_dtype);
        CHECK_2D(outputs[i], total_nrows_ep, ffn_size_shard);
      }
    }

    // 步骤 4: 准备 GEMM 参数
    using UnderlyingProblemShape = cute::Shape<int, int, int>;
    std::vector<UnderlyingProblemShape> problem_sizes;
    std::vector<void const *> ptr_A;
    std::vector<void const *> ptr_B;
    std::vector<void const *> ptr_C;
    std::vector<void *> ptr_D;
    std::vector<float *> ptr_alpha;
    std::vector<int32_t const *> ptr_gather_A;
    std::vector<int32_t const *> ptr_scatter_D;
    std::vector<ProblemSchedule> problem_schedules_arg;
    {
      int tile_M = cute::get<0>(op->get_runtime_gemm_hparams().tile_shape());
      auto problem_schedules =
          get_sorted_problem_schedule(sorted_splits_cpu, tp_env, nexperts_ep, tile_M, 4);

      int const N = ffn_size_shard;
      int const K = moe_args.hidden;
      int64_t const weight_bytes = 1LL * N * K * c10::elementSize(moe_args.input_dtype);

      for (std::size_t idx = 0; idx < problem_schedules.size(); ++idx) {
        auto const &problem_param = problem_schedules[idx];
        int Mi = problem_param.m_end - problem_param.m_start;
        if (Mi == 0) {
          continue;
        }
        for (std::size_t i = 0; i < num_weights_group; ++i) {
          problem_sizes.emplace_back(Mi, N, K);
          ptr_A.emplace_back(input_buffer.data_ptr());  // 指向 NVSHMEM 共享缓冲区
          ptr_B.emplace_back(
              ptr_offset(weights[i].data_ptr(), problem_param.expert_id * weight_bytes));
          ptr_C.emplace_back(nullptr);
          ptr_D.emplace_back(outputs[i].data_ptr());
          ptr_gather_A.emplace_back(
              static_cast<int32_t const *>(ptr_offset(
                  sorted_gather_index.data_ptr(), 1LL * sizeof(int32_t) * problem_param.m_start)));
          ptr_scatter_D.emplace_back(
              static_cast<int32_t const *>(ptr_offset(
                  sorted_scatter_index.data_ptr(),
                  1LL * sizeof(int32_t) * problem_param.m_start)));
          problem_schedules_arg.emplace_back(problem_param);

          if (output_scale.has_value()) {
            ptr_alpha.emplace_back(
                static_cast<float *>(output_scale->at(i).data_ptr()) + problem_param.expert_id);
          }
        }
      }
    }

    auto args = GemmGroupedAgScatterArguments{
        GemmGroupedV3Arguments{
            .problem_count = static_cast<int>(problem_sizes.size()),
            .alpha = 1.0f,
            .beta = 0.0f,
            .problem_sizes = problem_sizes.data(),
            .ptr_A = ptr_A.data(),
            .ptr_B = ptr_B.data(),
            .ptr_C = ptr_C.data(),
            .ptr_D = ptr_D.data(),
            .ptr_alpha = output_scale.has_value() ? ptr_alpha.data() : nullptr},
        static_cast<DistEnv>(tp_env),
        ntokens,
        moe_args.hidden,
        input_buffer.data_ptr(),  // NVSHMEM 共享缓冲区指针
        ptr_gather_A.data(),
        ptr_scatter_D.data(),
        problem_schedules_arg.data(),
    };

    int64_t workspace_size = op->get_workspace_size(args);
    lazy_init_buffer_tensor(&this->workspace_buffer, workspace_size);
    args.barrier_ptr = barrier_block.get();  // 设置屏障指针
    args.sm_margin = sm_margin;

    if (tp_env.nnodes > 1) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, this->fetch_remote_event));
    }
    // 步骤 5: 启动 GEMM kernel - Kernel 直接访问 NVSHMEM 共享缓冲区
    if (args.problem_count > 0) {
      op->run(args, workspace_buffer.data_ptr(), stream);
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, this->all_gather_event));
    // 确保所有节点都完成了 all-gather 操作，避免数据竞争
    nvshmemx_barrier_all_on_stream(stream);
    if (allgather_output.has_value()) {
      CHECK_INPUT(allgather_output.value(), moe_args.input_dtype);
      CHECK_2D(allgather_output.value(), ntokens, moe_args.hidden);
      CUDA_CHECK(cudaMemcpyAsync(
          allgather_output->data_ptr(),
          input_buffer.data_ptr(),
          allgather_output->nbytes(),
          cudaMemcpyDeviceToDevice,
          stream));
    }
    return outputs;
  }

 public:
  void
  clear_buffers() {
    this->input_buffer.fill_(0);
    this->sorted_splits.fill_(0);
    this->gather_index.fill_(0);
    this->sorted_scatter_index.fill_(0);
    this->sorted_gather_index.fill_(0);
    this->sorted_splits_cpu.assign(this->sorted_splits_cpu.size(), 0);
  }

  std::vector<torch::Tensor>
  forward_multiple_weights(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin) {
    return forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        c10::nullopt);
  }

  torch::Tensor
  forward(
      torch::Tensor inputs_shard,
      torch::Tensor weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<torch::Tensor> output_scale,
      c10::optional<torch::Tensor> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin) {
    c10::optional<std::vector<torch::Tensor>> output_scale_wrap;
    c10::optional<std::vector<torch::Tensor>> outputs_buf_wrap;

    if (output_scale.has_value()) {
      output_scale_wrap = {output_scale.value()};
    }

    if (outputs_buf.has_value()) {
      outputs_buf_wrap = {outputs_buf.value()};
    }

    auto outputs = forward_impl(
        std::move(inputs_shard),
        {weights},
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(output_scale_wrap),
        std::move(outputs_buf_wrap),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        c10::nullopt);
    return outputs[0];
  }

  std::vector<torch::Tensor>
  profiling(
      torch::Tensor inputs_shard,
      std::vector<torch::Tensor> weights,
      torch::Tensor splits_gpu,
      torch::Tensor scatter_index,
      c10::optional<std::vector<torch::Tensor>> output_scale,
      c10::optional<std::vector<torch::Tensor>> outputs_buf,
      c10::optional<torch::Tensor> allgather_output,
      bool fast_accum,
      int sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto meta = unify_type(this->get_gemm_meta(fast_accum));
    auto rt_conf = this->get_rt_conf();
    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto elapsed_tensor = torch::empty({}, inputs_shard.options().dtype(c10::ScalarType::Float));
    auto reduced_elapsed_tensor = elapsed_tensor.clone();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          // filter non-consistent hparams
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          auto cp_hparams = hparams;
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                inputs_shard,
                weights,
                splits_gpu,
                scatter_index,
                output_scale,
                outputs_buf,
                allgather_output,
                fast_accum,
                sm_margin,
                cp_hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          // Avoid GPU frequency adjustment
          nvshmemx_barrier_all_on_stream(stream);
          c10::cuda::stream_synchronize(stream);
          sleep(1);

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          elapsed_tensor.copy_(torch::full({}, avg_elapsed));

          nvshmemx_float_max_reduce_on_stream(
              NVSHMEM_TEAM_WORLD,
              static_cast<float *>(reduced_elapsed_tensor.data_ptr()),
              static_cast<float const *>(elapsed_tensor.data_ptr()),
              1,
              stream);

          float reduce_elapsed = reduced_elapsed_tensor.item().toFloat();
          ctx->add(meta, rt_conf, hparams, reduce_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);
    return this->forward_impl(
        std::move(inputs_shard),
        std::move(weights),
        std::move(splits_gpu),
        std::move(scatter_index),
        std::move(output_scale),
        std::move(outputs_buf),
        std::move(allgather_output),
        fast_accum,
        sm_margin,
        std::move(best_hparams));
  }
};

GemmGroupedV3AGScatterOp::GemmGroupedV3AGScatterOp(DistEnvTPWithEP tp_env_, MoeArguments moe_args)
    : impl_(new GemmGroupedV3AGScatterOpImpl(tp_env_, moe_args)) {}
GemmGroupedV3AGScatterOp::~GemmGroupedV3AGScatterOp() { delete impl_; }
void
GemmGroupedV3AGScatterOp::clear_buffers() {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3AGScatterOp::clear_buffers(): is not initialized";
  impl_->clear_buffers();
}
torch::Tensor
GemmGroupedV3AGScatterOp::forward(
    torch::Tensor inputs_shard,
    torch::Tensor weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<torch::Tensor> output_scale,
    c10::optional<torch::Tensor> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3AGScatterOp::forward(): is not initialized";
  return impl_->forward(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin);
}
std::vector<torch::Tensor>
GemmGroupedV3AGScatterOp::forward_multiple_weights(
    torch::Tensor inputs_shard,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<std::vector<torch::Tensor>> output_scale,
    c10::optional<std::vector<torch::Tensor>> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin) {
  FLUX_CHECK(impl_ != nullptr)
      << "GemmGroupedV3AGScatterOp::forward_multiple_weights(): is not initialized";
  return impl_->forward_multiple_weights(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin);
}
std::vector<torch::Tensor>
GemmGroupedV3AGScatterOp::profiling(
    torch::Tensor inputs_shard,
    std::vector<torch::Tensor> weights,
    torch::Tensor splits_gpu,
    torch::Tensor scatter_index,
    c10::optional<std::vector<torch::Tensor>> output_scale,
    c10::optional<std::vector<torch::Tensor>> outputs_buf,
    c10::optional<torch::Tensor> allgather_output,
    bool fast_accum,
    int sm_margin,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV3AGScatterOp::profiling(): is not initialized";
  return impl_->profiling(
      std::move(inputs_shard),
      std::move(weights),
      std::move(splits_gpu),
      std::move(scatter_index),
      std::move(output_scale),
      std::move(outputs_buf),
      std::move(allgather_output),
      fast_accum,
      sm_margin,
      std::move(opt_ctx));
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
