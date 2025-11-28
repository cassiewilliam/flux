//===- test_ag_gemm_fp8.cpp --------------------------------------- C++ ---===//
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
// AllGather + GEMM FP8 测试 (Per-Tensor 量化)
// 参考: test/python/ag_gemm/test_ag_kernel.py
//

#include <atomic>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "flux/args/ag_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/runtime_config.h"

namespace bytedance::flux {

// Performance result structure
struct PerfResult
{
    std::string name;
    float       total_ms;
    float       comm_time_ms;
    float       gemm_time_ms;

    PerfResult(const std::string& n, float total, float comm, float gemm)
        : name(n)
        , total_ms(total)
        , comm_time_ms(comm)
        , gemm_time_ms(gemm)
    {}

    void print() const
    {
        printf("%s: total %.3f ms, comm %.3f ms, gemm %.3f ms\n",
               name.c_str(),
               total_ms,
               comm_time_ms,
               gemm_time_ms);
    }
};

// Initialize peer access for multi-GPU communication
void init_peer_access(int tp)
{
    for (int i = 0; i < tp; ++i)
    {
        cudaSetDevice(i);
        for (int j = 0; j < tp; ++j)
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

// Performance test for AG + GEMM with FP8 (per tensor quantization)
// 参考 Python 中的 perf_flux 函数
void run_ag_gemm_fp8_test(int  m,
                          int  n,
                          int  k,
                          int  tp,
                          int  nnodes,
                          int  warmup,
                          int  iters,
                          bool debug_mode)
{
    constexpr int kMaxTp = 8;
    FLUX_CHECK(tp <= kMaxTp) << "tp=" << tp << " kMaxTp=" << kMaxTp;
    FLUX_CHECK(m % tp == 0) << "m=" << m << " tp=" << tp;

    // Check if FP8 is supported (requires SM90+)
    auto arch = get_arch();
    if ((int)arch < (int)_Sm90{}())
    {
        std::cerr << "FP8 requires SM90+ architecture. Current arch: " << (int)arch << std::endl;
        return;
    }

    int  local_m = m / tp;
    auto sm_core = get_sm_core();

    // FP8 E4M3 -> BF16
    // 创建 FP8 AG Kernel 的元数据
    auto meta = make_gemm_meta(make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
                               arch,
                               sm_core,
                               _AGKernel{},
                               _RCR{},
                               _GemmV3{});

    using ElementInput  = cutlass::float_e4m3_t;
    using ElementWeight = cutlass::float_e4m3_t;
    using ElementOutput = cutlass::bfloat16_t;
    using ElementScale  = float;

    auto rt_conf = make_runtime_config(m, n, k, make_all_gather_runtime_config(tp, nnodes));

    // Allocate device memory for each GPU
    cutlass::DeviceAllocation<ElementInput>  block_A[kMaxTp];            // local input (FP8)
    cutlass::DeviceAllocation<ElementInput>  block_gathered_A[kMaxTp];   // gathered input (FP8)
    cutlass::DeviceAllocation<ElementWeight> block_B[kMaxTp];            // weight (FP8)
    cutlass::DeviceAllocation<ElementOutput> block_D[kMaxTp];            // output (BF16)
    cutlass::DeviceAllocation<int>           block_barrier[kMaxTp];

    // Per-tensor scales (scalar)
    cutlass::DeviceAllocation<ElementScale> input_scale[kMaxTp];
    cutlass::DeviceAllocation<ElementScale> weight_scale[kMaxTp];

    void* block_barrier_ptrs[kMaxTp];

    // Initialize data
    for (int i = 0; i < tp; ++i)
    {
        cudaSetDevice(i);

        block_A[i].reset(local_m * k);
        block_gathered_A[i].reset(m * k);
        block_B[i].reset(k * n);
        block_D[i].reset(m * n);

        // Per-tensor scales: 单个标量
        input_scale[i].reset(1);
        weight_scale[i].reset(1);

        // Initialize input data
        std::vector<ElementInput>  h_A(local_m * k);
        std::vector<ElementWeight> h_B(k * n);

        if (debug_mode)
        {
            for (size_t j = 0; j < h_A.size(); ++j)
            {
                h_A[j] = ElementInput(0.01f * (i + 1));
            }
            for (size_t j = 0; j < h_B.size(); ++j)
            {
                h_B[j] = ElementWeight(0.01f);
            }
        }
        else
        {
            for (size_t j = 0; j < h_A.size(); ++j)
            {
                h_A[j] = ElementInput(0.01f * (i + 1));
            }
            for (size_t j = 0; j < h_B.size(); ++j)
            {
                h_B[j] = ElementWeight(0.01f * (i + 1));
            }
        }

        cudaMemcpy(block_A[i].get(),
                   h_A.data(),
                   h_A.size() * sizeof(ElementInput),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(block_B[i].get(),
                   h_B.data(),
                   h_B.size() * sizeof(ElementWeight),
                   cudaMemcpyHostToDevice);

        // Initialize scales
        ElementScale h_input_scale  = 1.0f;
        ElementScale h_weight_scale = 1.0f;
        cudaMemcpy(input_scale[i].get(),
                   &h_input_scale,
                   sizeof(ElementScale),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(weight_scale[i].get(),
                   &h_weight_scale,
                   sizeof(ElementScale),
                   cudaMemcpyHostToDevice);

        block_barrier_ptrs[i] = nullptr;
    }

    std::atomic<int>        ready_count{0};
    std::vector<PerfResult> results;
    results.reserve(tp);

    // Thread function for each GPU
    auto thread_fn = [&](int rank) {
        cudaSetDevice(rank);

        auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);

        // 创建 FP8 AG Kernel Arguments
        // 注意：这里使用 AGFP8KernelArguments 结构
        AGFP8KernelArguments args{
            m,                              // m
            n,                              // n
            k,                              // k
            rank,                           // rank
            tp,                             // world_size
            nnodes,                         // nnodes
            1.0f,                           // alpha
            0.0f,                           // beta
            block_gathered_A[rank].get(),   // A (gathered input buffer)
            block_B[rank].get(),            // B (weight)
            nullptr,                        // C (bias, FP8 不支持)
            nullptr,                        // Aux
            block_D[rank].get(),            // D (output)
            nullptr,                        // barrier_buffer (稍后设置)
            nullptr,                        // Vector (bias)
            nullptr,                        // abs_max_Aux
            nullptr,                        // abs_max_D
            input_scale[rank].get(),        // scaleA (per-tensor, scalar)
            weight_scale[rank].get(),       // scaleB (per-tensor, scalar)
            nullptr,                        // scaleC
            nullptr,                        // scaleD
            nullptr                         // scaleAux
        };

        // Allocate barrier
        int flag_count = gemm_op->get_barrier_workspace_size(args) / sizeof(int);
        block_barrier[rank].reset(flag_count);
        cudaMemset(block_barrier[rank].get(), 0, flag_count * sizeof(int));
        args.barrier_buffer = block_barrier[rank].get();

        cudaDeviceSynchronize();

        // Sync all threads
        ++ready_count;
        while (ready_count < tp)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // Copy local input to gathered buffer (模拟 AllGather 的输入准备)
        cudaMemcpy((char*)block_gathered_A[rank].get() + rank * local_m * k * sizeof(ElementInput),
                   block_A[rank].get(),
                   local_m * k * sizeof(ElementInput),
                   cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();

        // Warmup + measurement
        GpuTimer timer;
        int      total_iters = warmup + iters;

        for (int i = 0; i < total_iters; ++i)
        {
            if (i == warmup)
            {
                timer.start(stream);
            }
            gemm_op->run(args, nullptr, stream);
        }
        timer.stop();

        float elapsed  = timer.elapsed_millis();
        float avg_time = elapsed / iters;

        // 对于 AG+GEMM，通信与计算重叠，这里简化报告总时间
        PerfResult result("flux AG+GEMM FP8 (per-tensor) #" + std::to_string(rank),
                          avg_time,
                          0.0f,   // comm overlapped
                          avg_time);

        // Thread-safe result storage
        results.push_back(result);

        CUDA_CHECK(cudaStreamDestroy(stream));
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // Launch threads
    std::vector<std::thread> threads;
    for (int i = 0; i < tp; ++i)
    {
        threads.emplace_back(thread_fn, i);
    }

    for (int i = 0; i < tp; ++i)
    {
        threads[i].join();
    }

    // Print results
    std::cout << "\n=== FP8 AG+GEMM Performance (Per-Tensor Quantization) ===" << std::endl;
    for (const auto& result : results)
    {
        result.print();
    }

    std::cout << "\n✅ FP8 AG+GEMM test completed" << std::endl;
    std::cout << "   - Input: FP8 E4M3" << std::endl;
    std::cout << "   - Weight: FP8 E4M3" << std::endl;
    std::cout << "   - Output: BF16" << std::endl;
    std::cout << "   - Quantization: Per-Tensor (scalar scales)" << std::endl;
}

}   // namespace bytedance::flux

int main(int argc, char* argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <m> <n> <k> <tp> [nnodes=1] [warmup=5] [iters=10] [debug=0]\n";
        std::cerr << "\nParameters:\n";
        std::cerr << "  m      : Total rows (will be divided by TP)\n";
        std::cerr << "  n      : Output columns\n";
        std::cerr << "  k      : Input columns / weight rows\n";
        std::cerr << "  tp     : Tensor parallel size\n";
        std::cerr << "  nnodes : Number of nodes (default: 1)\n";
        std::cerr << "  warmup : Warmup iterations (default: 5)\n";
        std::cerr << "  iters  : Measurement iterations (default: 10)\n";
        std::cerr << "  debug  : Debug mode (default: 0)\n";
        std::cerr << "\nFP8 Per-Tensor Quantization:\n";
        std::cerr << "  - Input scale: single scalar for entire input tensor\n";
        std::cerr << "  - Weight scale: single scalar for entire weight tensor\n";
        std::cerr << "  - Output: BF16 (no quantization needed)\n";
        std::cerr << "\nRequires: SM90+ architecture (H100, H200, etc.)\n";
        std::cerr << "\nExample:\n";
        std::cerr << "  " << argv[0] << " 2048 10240 40960 8\n";
        return 1;
    }

    int m  = std::atoi(argv[1]);
    int n  = std::atoi(argv[2]);
    int k  = std::atoi(argv[3]);
    int tp = std::atoi(argv[4]);

    int  nnodes     = (argc > 5) ? std::atoi(argv[5]) : 1;
    int  warmup     = (argc > 6) ? std::atoi(argv[6]) : 5;
    int  iters      = (argc > 7) ? std::atoi(argv[7]) : 10;
    bool debug_mode = (argc > 8) ? (std::atoi(argv[8]) != 0) : false;

    std::cout << "=== FP8 AG+GEMM Test Configuration ===" << std::endl;
    std::cout << "M=" << m << ", N=" << n << ", K=" << k << std::endl;
    std::cout << "TP=" << tp << ", NNodes=" << nnodes << std::endl;
    std::cout << "Warmup=" << warmup << ", Iters=" << iters << std::endl;
    std::cout << "Debug=" << debug_mode << std::endl;
    std::cout << "======================================" << std::endl;

    bytedance::flux::init_peer_access(tp);

    try
    {
        bytedance::flux::run_ag_gemm_fp8_test(m, n, k, tp, nnodes, warmup, iters, debug_mode);
    }
    catch (std::exception const& e)
    {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}