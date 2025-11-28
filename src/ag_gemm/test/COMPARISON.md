# Python 版本 vs C++ 版本对比

本文档说明 `test_ag_gemm.cpp` 与参考的 `test/python/gemm_rs/test_gemm_rs.py` 之间的对应关系和主要差异。

## 架构对比

### Python 版本 (test_gemm_rs.py)

```python
# 使用 torchrun 启动多进程
torchrun --nproc_per_node=8 test_gemm_rs.py

# 主要组件：
1. perf_torch()    - PyTorch 原生实现
2. perf_flux()     - Flux 实现
3. perf_triton()   - Triton 实现 (已在 C++ 版本中移除)
```

### C++ 版本 (test_ag_gemm.cpp)

```cpp
// 单进程多线程
./test_ag_gemm <args>

// 主要组件：
1. run_ag_gemm()   - Flux C++ 实现
2. thread_fn()     - 每个 GPU 的线程函数
```

## 功能对应表

| 功能 | Python (test_gemm_rs.py) | C++ (test_ag_gemm.cpp) | 说明 |
|-----|-------------------------|------------------------|------|
| **操作类型** | GemmRS (Reduce Scatter) | AG Gemm (All Gather) | 不同的通信模式 |
| **并行方式** | 多进程 (torchrun) | 多线程 (std::thread) | 实现方式不同 |
| **Flux 实现** | ✅ perf_flux() | ✅ run_ag_gemm() | 核心测试 |
| **PyTorch 基线** | ✅ perf_torch() | ❌ 未实现 | C++ 版本专注于 Flux |
| **Triton 实现** | ✅ perf_triton() | ❌ 已移除 | 按需求移除 |
| **数据类型** | 多种 (fp16/bf16/fp8/int8) | fp16 | C++ 版本当前仅 fp16 |
| **Bias 支持** | ✅ --has_bias | ✅ has_bias 参数 | 两者都支持 |
| **转置权重** | ✅ --transpose_weight | ✅ transpose_weight | 两者都支持 |
| **性能分析** | ✅ --profile | ❌ 未实现 | Python 使用 torch.profiler |
| **调试模式** | ✅ --debug | ✅ debug 参数 | 两者都支持 |
| **正确性检查** | ✅ torch_allclose | ✅ (简化版) | 两者都验证结果 |

## 代码结构对比

### 1. 初始化分布式环境

**Python:**
```python
TP_GROUP = initialize_distributed()
RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()
```

**C++:**
```cpp
init_peer_access(tp);  // 启用 GPU P2P 访问
// rank 和 world_size 在线程函数中管理
```

### 2. 性能测试结构

**Python:**
```python
class PerfResult:
    def __init__(self, name, output, gemm_time_ms, comm_time_ms):
        self.name = name
        self.output = output
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms
```

**C++:**
```cpp
struct PerfResult {
  std::string name;
  float gemm_time_ms;
  float comm_time_ms;
  float total_ms;
  
  void print() const;
};
```

### 3. Flux 操作调用

**Python (GemmRS):**
```python
gemm_rs_op = flux.GemmRS(
    TP_GROUP,
    NNODES,
    (M + 1024 - 1) // 1024 * 1024,
    N,
    input.dtype,
    output_dtype,
    transpose_weight=transpose_weight,
    fuse_reduction=fuse_reduction,
    ring_reduction=ring_reduction,
)
output = gemm_rs_op.forward(input, weight, bias=bias, ...)
```

**C++ (AG Gemm):**
```cpp
auto meta = make_gemm_meta(
    _FP16{}, arch, sm_core, _AllGather{}, _RCR{},
    ..., make_all_gather_meta(_IntraNode{})
);
auto rt_conf = make_runtime_config(
    m, n, k, make_all_gather_runtime_config(tp, nnodes)
);
auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);

AGKernelArguments args{m, n, k, rank, tp, nnodes, ...};
gemm_op->run(args, nullptr, stream);
```

### 4. 性能计时

**Python:**
```python
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

for i in range(total_iters):
    start_events[i].record()
    output = gemm_rs_op.forward(...)
    end_events[i].record()

# 计算平均时间
```

**C++:**
```cpp
GpuTimer gemm_timer;
for (int i = 0; i < total_iters; ++i) {
  if (i == warmup) {
    gemm_timer.start(stream);
  }
  gemm_op->run(args, nullptr, stream);
}
gemm_timer.stop();
float avg_time = gemm_timer.elapsed_millis() / iters;
```

## 参数映射

### Python 命令行

```bash
torchrun --nproc_per_node=8 test_gemm_rs.py 2048 10240 40960 \
    --warmup 5 \
    --iters 100 \
    --dtype bfloat16 \
    --transpose_weight \
    --has_bias \
    --debug
```

### C++ 对应命令

```bash
./test_ag_gemm 2048 10240 40960 8 1 5 100 1 1 1
#              M    N     K     tp nnodes warmup iters transpose bias debug
```

## 主要差异说明

### 1. ✅ 已实现的功能

- ✅ 基本 GEMM 操作测试
- ✅ 性能测量（预热 + 迭代）
- ✅ 多 GPU 支持（通过多线程）
- ✅ Bias 支持
- ✅ 权重转置支持
- ✅ 调试模式
- ✅ 命令行参数配置

### 2. ❌ 已移除的功能 (按需求)

- ❌ Triton 实现对比 - **已按需求移除**
- ❌ PyTorch 基线对比 - 简化为仅测试 Flux
- ❌ 多数据类型支持 - 当前仅 fp16

### 3. 📝 实现差异

| 方面 | Python | C++ |
|-----|--------|-----|
| 并行模型 | 多进程 (torchrun) | 多线程 (std::thread) |
| 通信操作 | Reduce Scatter | All Gather |
| 同步方式 | torch.distributed.barrier | std::atomic + sleep |
| 内存管理 | torch.Tensor | cutlass::DeviceAllocation |
| 计时方式 | torch.cuda.Event | GpuTimer (CUDA events) |

## 使用建议

### 何时使用 Python 版本

- 需要与 PyTorch 基线对比
- 需要测试多种数据类型
- 需要 Triton 实现对比
- 需要详细的 profiling 信息
- 快速原型开发

### 何时使用 C++ 版本

- 纯 Flux 性能测试
- 集成到 C++ 测试套件
- 不依赖 Python 环境
- 需要更精确的性能测量
- 生产环境部署前验证

## 扩展建议

如果需要扩展 C++ 版本以更接近 Python 版本的功能：

1. **添加 PyTorch 基线**：使用 LibTorch 实现 torch.matmul + reduce_scatter
2. **多数据类型支持**：添加模板或参数控制数据类型
3. **性能分析**：集成 NVIDIA Nsight 或 CUPTI
4. **多节点支持**：集成 NCCL 或其他通信库
5. **正确性验证**：实现更完整的结果对比

## 注意事项

1. **操作类型不同**：Python 版本测试 GemmRS（Reduce Scatter），C++ 版本测试 AG Gemm（All Gather）
2. **通信模式**：两者的通信模式不同，性能特征也不同
3. **环境要求**：C++ 版本需要多 GPU 在同一节点上，Python 版本可跨节点
4. **编译需求**：C++ 版本需要编译，Python 版本可直接运行

## 总结

C++ 版本是参照 Python 版本设计的轻量级测试实现，专注于：
- ✅ Flux C++ API 的正确性验证
- ✅ 性能基准测试
- ✅ 去除 Triton 依赖（按需求）
- ✅ 简化的测试流程

适合在不需要 Python 环境的情况下进行快速的 Flux 性能测试和验证。

