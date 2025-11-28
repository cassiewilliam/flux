# AG+GEMM C++ 测试

这个测试是参照 `test/python/gemm_rs/test_gemm_rs.py` 实现的 C++ 版本，用于测试 AllGather + GEMM 操作，已去除 Triton 依赖。

## 编译

确保在构建 Flux 时启用了 `BUILD_TEST` 选项：

```bash
cd /Users/min.yang/learncode/flux
mkdir -p build && cd build
cmake .. -DBUILD_TEST=ON
make test_ag_gemm
```

## 使用方法

### 基本用法

```bash
./build/src/ag_gemm/test/test_ag_gemm <m> <n> <k> <tp>
```

### 完整参数

```bash
./test_ag_gemm <m> <n> <k> <tp> [nnodes] [warmup] [iters] [transpose_weight] [has_bias] [debug]
```

### 参数说明

- `m`: 输入矩阵的行数（必须能被 tp 整除）
- `n`: 权重矩阵的列数（输出列数）
- `k`: 输入矩阵的列数 / 权重矩阵的行数
- `tp`: 张量并行大小（world size，GPU 数量）
- `nnodes`: 节点数量（默认：1）
- `warmup`: 预热迭代次数（默认：5）
- `iters`: 性能测量迭代次数（默认：10）
- `transpose_weight`: 是否转置权重矩阵（默认：0）
- `has_bias`: 是否包含偏置（默认：0）
- `debug`: 调试模式，使用简单值便于验证（默认：0）

## 示例

### 示例 1: 基本测试（8 GPU，与 Python 版本对应）

```bash
./test_ag_gemm 2048 10240 40960 8
```

### 示例 2: 带偏置的测试

```bash
./test_ag_gemm 2048 10240 40960 8 1 5 10 0 1
```

### 示例 3: 调试模式

```bash
./test_ag_gemm 2048 10240 40960 8 1 2 2 0 0 1
```

### 示例 4: 更多迭代用于精确性能测量

```bash
./test_ag_gemm 2048 10240 40960 8 1 10 100
```

## 输出

测试会输出每个 GPU rank 的性能数据：

```
=== AG+GEMM Test Configuration ===
M=2048, N=10240, K=40960
TP=8, NNodes=1
Warmup=5, Iterations=10
...
===================================
flux AG+GEMM #0: gemm 2.345 ms, comm 0.000 ms, total 2.345 ms
flux AG+GEMM #1: gemm 2.347 ms, comm 0.000 ms, total 2.347 ms
...
✅ AG+GEMM test completed successfully
```

## 与 Python 版本的对应关系

| Python 参数 | C++ 参数位置 | 说明 |
|------------|------------|------|
| `M` | argv[1] | 矩阵 M 维度 |
| `N` | argv[2] | 矩阵 N 维度 |
| `K` | argv[3] | 矩阵 K 维度 |
| `--warmup` | argv[6] | 预热迭代 |
| `--iters` | argv[7] | 测试迭代 |
| `--transpose_weight` | argv[8] | 转置权重 |
| `--has_bias` | argv[9] | 包含偏置 |
| `--debug` | argv[10] | 调试模式 |

## 主要差异

1. **去除 Triton**: C++ 版本只测试 Flux 实现，不包含 Triton 比较
2. **简化通信测量**: AG+GEMM 操作中通信与计算重叠，因此报告总时间
3. **多线程而非多进程**: 使用 C++ 线程而不是 torchrun 的多进程
4. **FP16 数据类型**: 当前实现使用 FP16，Python 版本支持多种数据类型

## 注意事项

1. 确保有足够的 GPU（至少等于 `tp` 参数）
2. GPU 之间需要支持 P2P 访问
3. M 必须能被 tp 整除
4. 对于多节点测试（nnodes > 1），tp 必须能被 nnodes 整除

## 故障排除

如果遇到编译错误，确保：
- CUDA 正确安装且版本兼容
- Cutlass 库已正确配置
- CMake 配置中启用了 `BUILD_TEST`

如果遇到运行时错误：
- 检查 GPU 数量是否足够
- 检查 P2P 访问是否可用：`nvidia-smi topo -m`
- 尝试使用调试模式进行简单测试

