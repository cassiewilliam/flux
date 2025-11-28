//===- test_ag_gemm_pure_cpp.cpp ---------------------------------- C++ ---===//
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
//
// 纯 C++ AllGather + GEMM 测试 (无 PyTorch 依赖)
//
// 调用链路:
// 1. 直接使用 OpRegistry 获取 GEMM operator
// 2. 手动构造 AGKernelArguments / AGFP8KernelArguments
// 3. 使用 CUDA 原生内存管理
// 4. 模拟 AllGather 通信
// 5. 执行 GEMM 计算
//

#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "flux/args/ag_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"

namespace bytedance::flux::test {

// 性能结果
struct PerfResult
{
    int   rank;
    float avg_ms;
    float min_ms;
    float max_ms;
    float tflops;

    void print() const
    {
        printf("  Rank %d: avg %.3f ms, min %.3f ms, max %.3f ms, %.2f TFLOPS\n",
               rank,
               avg_ms,
               min_ms,
               max_ms,
               tflops);
    }
};

// 初始化 peer access
void init_peer_access(int world_size)
{
    for (int i = 0; i < world_size; ++i)
    {
        cudaSetDevice(i);
        for (int j = 0; j < world_size; ++j)
        {
            if (j != i)
            {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled)
                {
                    CUDA_CHECK(err);
                }
            }
        }
    }
    cudaSetDevice(0);
}

// FP8 测试
void test_fp8_ag_gemm(int  m,
                      int  n,
                      int  k,
                      int  world_size,
                      int  nnodes,
                      int  warmup,
                      int  iters,
                      bool verify)
{

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "测试 FP8 AllGather + GEMM (纯 C++, 无 PyTorch)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    auto arch = get_arch();
    if ((int)arch < (int)_Sm90{}())
    {
        std::cerr << "❌ FP8 需要 SM90+ 架构" << std::endl;
        return;
    }

    FLUX_CHECK(m % world_size == 0);
    int local_m = m / world_size;

    std::cout << "\n配置:" << std::endl;
    std::cout << "  M=" << m << " (local_m=" << local_m << "), N=" << n << ", K=" << k << std::endl;
    std::cout << "  world_size=" << world_size << ", nnodes=" << nnodes << std::endl;

    auto sm_core = get_sm_core();

    // FP8 E4M3 -> BF16
    auto meta = make_gemm_meta(make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}, _FP32{}),
                               arch,
                               sm_core,
                               _AGKernel{},
                               _RCR{},
                               _GemmV3{});

    auto rt_conf = make_runtime_config(m, n, k, make_all_gather_runtime_config(world_size, nnodes));

    std::cout << "\nGEMM Meta: " << meta << std::endl;

    using ElementInput  = cutlass::float_e4m3_t;
    using ElementWeight = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementScale  = float;

    constexpr int kMaxWorldSize = 8;

    cutlass::DeviceAllocation<ElementInput>  block_local_A[kMaxWorldSize];
    cutlass::DeviceAllocation<ElementInput>  block_gathered_A[kMaxWorldSize];
    cutlass::DeviceAllocation<ElementWeight> block_B[kMaxWorldSize];
    cutlass::DeviceAllocation<ElementOutput> block_D[kMaxWorldSize];
    cutlass::DeviceAllocation<int>           block_barrier[kMaxWorldSize];
    cutlass::DeviceAllocation<ElementScale>  input_scale[kMaxWorldSize];
    cutlass::DeviceAllocation<ElementScale>  weight_scale[kMaxWorldSize];

    std::atomic<int>        ready_count{0};
    std::vector<PerfResult> results;

    auto thread_fn = [&](int rank) {
        try
        {
            CUDA_CHECK(cudaSetDevice(rank));

            size_t size_local_A    = static_cast<size_t>(local_m) * k;
            size_t size_gathered_A = static_cast<size_t>(m) * k;
            size_t size_B          = static_cast<size_t>(k) * n;
            size_t size_D          = static_cast<size_t>(m) * n;

            block_local_A[rank].reset(size_local_A);
            block_gathered_A[rank].reset(size_gathered_A);
            block_B[rank].reset(size_B);
            block_D[rank].reset(size_D);
            input_scale[rank].reset(1);
            weight_scale[rank].reset(1);

            // 初始化数据
            std::vector<ElementInput>  h_A(size_local_A);
            std::vector<ElementWeight> h_B(size_B);

            float val_A = 0.01f * (rank + 1);
            float val_B = 0.01f;

            for (size_t i = 0; i < h_A.size(); ++i)
            {
                h_A[i] = ElementInput(val_A);
            }
            for (size_t i = 0; i < h_B.size(); ++i)
            {
                h_B[i] = ElementWeight(val_B);
            }

            CUDA_CHECK(cudaMemcpy(block_local_A[rank].get(),
                                  h_A.data(),
                                  h_A.size() * sizeof(ElementInput),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(block_B[rank].get(),
                                  h_B.data(),
                                  h_B.size() * sizeof(ElementWeight),
                                  cudaMemcpyHostToDevice));

            ElementScale h_scale = 1.0f;
            CUDA_CHECK(cudaMemcpy(input_scale[rank].get(),
                                  &h_scale,
                                  sizeof(ElementScale),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(weight_scale[rank].get(),
                                  &h_scale,
                                  sizeof(ElementScale),
                                  cudaMemcpyHostToDevice));

            std::cout << "[Rank " << rank << "] 数据初始化完成 (FP8)" << std::endl;

            auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
            FLUX_CHECK(gemm_op != nullptr);

            AGFP8KernelArguments args{.m              = m,
                                      .n              = n,
                                      .k              = k,
                                      .rank           = rank,
                                      .world_size     = world_size,
                                      .nnodes         = nnodes,
                                      .alpha          = 1.0f,
                                      .beta           = 0.0f,
                                      .A              = block_gathered_A[rank].get(),
                                      .B              = block_B[rank].get(),
                                      .C              = nullptr,
                                      .Aux            = nullptr,
                                      .D              = block_D[rank].get(),
                                      .barrier_buffer = nullptr,
                                      .Vector         = nullptr,
                                      .abs_max_Aux    = nullptr,
                                      .abs_max_D      = nullptr,
                                      .scaleA         = input_scale[rank].get(),
                                      .scaleB         = weight_scale[rank].get(),
                                      .scaleC         = nullptr,
                                      .scaleD         = nullptr,
                                      .scaleAux       = nullptr};

            size_t barrier_size = gemm_op->get_barrier_workspace_size(args);
            if (barrier_size > 0)
            {
                int flag_count = barrier_size / sizeof(int);
                block_barrier[rank].reset(flag_count);
                CUDA_CHECK(cudaMemset(block_barrier[rank].get(), 0, barrier_size));
                args.barrier_buffer = block_barrier[rank].get();
            }

            size_t                             workspace_size = gemm_op->get_workspace_size(args);
            cutlass::DeviceAllocation<uint8_t> workspace;
            void*                              workspace_ptr = nullptr;
            if (workspace_size > 0)
            {
                workspace.reset(workspace_size);
                workspace_ptr = workspace.get();
            }

            CUDA_CHECK(cudaDeviceSynchronize());

            ++ready_count;
            while (ready_count < world_size)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }

            cudaStream_t stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            // 模拟 AllGather
            size_t offset    = static_cast<size_t>(rank) * local_m * k * sizeof(ElementInput);
            size_t copy_size = static_cast<size_t>(local_m) * k * sizeof(ElementInput);
            CUDA_CHECK(cudaMemcpy((char*)block_gathered_A[rank].get() + offset,
                                  block_local_A[rank].get(),
                                  copy_size,
                                  cudaMemcpyDeviceToDevice));

            int                total_iters = warmup + iters;
            std::vector<float> iter_times;

            for (int i = 0; i < total_iters; ++i)
            {
                cudaEvent_t start, stop;
                if (i >= warmup)
                {
                    CUDA_CHECK(cudaEventCreate(&start));
                    CUDA_CHECK(cudaEventCreate(&stop));
                    CUDA_CHECK(cudaEventRecord(start, stream));
                }

                gemm_op->run(args, workspace_ptr, stream);

                if (i >= warmup)
                {
                    CUDA_CHECK(cudaEventRecord(stop, stream));
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float elapsed_ms = 0;
                    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
                    iter_times.push_back(elapsed_ms);
                    CUDA_CHECK(cudaEventDestroy(start));
                    CUDA_CHECK(cudaEventDestroy(stop));
                }
            }

            float sum = 0, min_time = iter_times[0], max_time = iter_times[0];
            for (float t : iter_times)
            {
                sum += t;
                min_time = std::min(min_time, t);
                max_time = std::max(max_time, t);
            }
            float avg_time = sum / iters;

            double flops  = 2.0 * m * n * k;
            float  tflops = (flops * 1e-12) / (avg_time * 1e-3);

            PerfResult result{rank, avg_time, min_time, max_time, tflops};
            results.push_back(result);

            std::cout << "[Rank " << rank << "] 完成 (FP8): " << tflops << " TFLOPS" << std::endl;

            CUDA_CHECK(cudaStreamDestroy(stream));
        }
        catch (const std::exception& e)
        {
            std::cerr << "❌ [Rank " << rank << "] 异常: " << e.what() << std::endl;
            throw;
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < world_size; ++i)
    {
        threads.emplace_back(thread_fn, i);
    }

    for (int i = 0; i < world_size; ++i)
    {
        threads[i].join();
    }

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "性能结果 (FP8)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    float total_tflops = 0;
    for (const auto& result : results)
    {
        result.print();
        total_tflops += result.tflops;
    }

    std::cout << "\n平均性能: " << (total_tflops / world_size) << " TFLOPS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

}   // namespace bytedance::flux::test

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <m> <n> <k> <world_size> [nnodes=1] [warmup=5] [iters=10] [verify=1]\n";
        std::cerr << "\n纯 C++ 实现 (无 PyTorch 依赖):\n";
        std::cerr << "  - 直接使用 OpRegistry 获取 GEMM operator\n";
        std::cerr << "  - CUDA 原生内存管理\n";
        std::cerr << "  - 模拟 AllGather 通信\n";
        std::cerr << "\n示例:\n";
        std::cerr << "  " << argv[0] << " 2048 4096 4096 4\n";
        std::cerr << "  " << argv[0] << " 4096 8192 8192 8 1 10 20 1\n";
        return 1;
    }

    int  m          = std::atoi(argv[1]);
    int  n          = std::atoi(argv[2]);
    int  k          = std::atoi(argv[3]);
    int  world_size = std::atoi(argv[4]);
    int  nnodes     = (argc > 5) ? std::atoi(argv[5]) : 1;
    int  warmup     = (argc > 6) ? std::atoi(argv[6]) : 5;
    int  iters      = (argc > 7) ? std::atoi(argv[7]) : 10;
    bool verify     = (argc > 8) ? (std::atoi(argv[8]) != 0) : true;

    std::cout << "========================================" << std::endl;
    std::cout << "AllGather + GEMM 纯 C++ 测试" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "特点:" << std::endl;
    std::cout << "  ✓ 无 PyTorch 依赖" << std::endl;
    std::cout << "  ✓ 直接使用 OpRegistry + CUTLASS" << std::endl;
    std::cout << "  ✓ CUDA 原生内存管理" << std::endl;
    std::cout << "========================================\n" << std::endl;

    bytedance::flux::test::init_peer_access(world_size);

    try
    {
        // 测试 BF16
        bytedance::flux::test::test_fp8_ag_gemm(m, n, k, world_size, nnodes, warmup, iters, verify);
    }
    catch (const std::exception& e)
    {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n✅ 所有测试完成 (纯 C++, 无 PyTorch 依赖)" << std::endl;
    return 0;
}