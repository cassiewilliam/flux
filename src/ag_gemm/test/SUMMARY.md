# 测试文件总结

本文档总结了为 AG+GEMM 创建的 C++ 测试实现。

## 📁 创建的文件清单

### 1. 核心测试文件

#### `test_ag_gemm.cpp`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/test_ag_gemm.cpp`
- **类型**: C++ 源文件
- **大小**: ~400 行
- **功能**: 
  - AG+GEMM 操作的性能测试
  - 多 GPU 并行测试（多线程）
  - 性能计时和结果输出
  - 可选的调试模式
- **参照**: `test/python/gemm_rs/test_gemm_rs.py`
- **主要改动**: 
  - ❌ 移除了 Triton 相关代码
  - ❌ 移除了 PyTorch 基线对比
  - ✅ 保留了核心 Flux 测试逻辑
  - ✅ 实现了类似的性能测量

### 2. 构建配置

#### `CMakeLists.txt`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/CMakeLists.txt`
- **类型**: CMake 配置
- **功能**: 定义测试可执行文件的编译规则
- **内容**:
  ```cmake
  add_executable(test_ag_gemm test_ag_gemm.cpp)
  target_link_libraries(test_ag_gemm PUBLIC flux_cuda)
  ```

#### 更新的 `src/ag_gemm/CMakeLists.txt`
- **修改**: 添加了测试子目录
- **新增内容**:
  ```cmake
  if (BUILD_TEST)
    add_subdirectory(test)
  endif()
  ```

### 3. 辅助脚本

#### `build_test.sh`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/build_test.sh`
- **类型**: Bash 脚本
- **权限**: 可执行 (chmod +x)
- **功能**: 自动化编译流程
- **使用**:
  ```bash
  cd /Users/min.yang/learncode/flux/src/ag_gemm/test
  ./build_test.sh
  ```

#### `run_test.sh`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/run_test.sh`
- **类型**: Bash 脚本
- **权限**: 可执行 (chmod +x)
- **功能**: 便捷运行测试，支持默认参数
- **使用**:
  ```bash
  ./run_test.sh [M] [N] [K] [TP] [...]
  # 或使用默认值
  ./run_test.sh
  ```

### 4. 文档

#### `README.md`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/README.md`
- **类型**: Markdown 文档
- **内容**:
  - 编译说明
  - 使用方法
  - 参数详解
  - 示例命令
  - 故障排除

#### `COMPARISON.md`
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/COMPARISON.md`
- **类型**: Markdown 文档
- **内容**:
  - Python vs C++ 版本对比
  - 功能对应表
  - 代码结构对比
  - 参数映射
  - 实现差异说明

#### `SUMMARY.md` (本文档)
- **位置**: `/Users/min.yang/learncode/flux/src/ag_gemm/test/SUMMARY.md`
- **类型**: Markdown 文档
- **内容**: 所有创建文件的总结

## 📊 文件结构树

```
src/ag_gemm/
├── CMakeLists.txt          (已更新 - 添加测试子目录)
├── test/                   (新建目录)
│   ├── test_ag_gemm.cpp   (主测试文件)
│   ├── CMakeLists.txt     (测试编译配置)
│   ├── build_test.sh      (编译脚本)
│   ├── run_test.sh        (运行脚本)
│   ├── README.md          (使用文档)
│   ├── COMPARISON.md      (对比文档)
│   └── SUMMARY.md         (总结文档 - 本文件)
└── [其他现有文件...]
```

## 🚀 快速开始

### 步骤 1: 编译

```bash
cd /Users/min.yang/learncode/flux/src/ag_gemm/test
./build_test.sh
```

### 步骤 2: 运行测试

```bash
# 使用默认参数（M=2048, N=10240, K=40960, TP=8）
./run_test.sh

# 或指定参数
./run_test.sh 2048 10240 40960 8

# 完整参数
./run_test.sh 2048 10240 40960 8 1 5 10 0 1 0
#             M    N     K     TP nnodes warmup iters trans bias debug
```

### 步骤 3: 查看结果

输出示例：
```
=== AG+GEMM Test Configuration ===
M=2048, N=10240, K=40960
TP=8, NNodes=1
...
flux AG+GEMM #0: gemm 2.345 ms, comm 0.000 ms, total 2.345 ms
...
✅ AG+GEMM test completed successfully
```

## 📝 与 Python 版本的对应

| Python 命令 | C++ 等效命令 |
|------------|-------------|
| `torchrun --nproc_per_node=8 test_gemm_rs.py 2048 10240 40960` | `./test_ag_gemm 2048 10240 40960 8` |
| `test_gemm_rs.py ... --warmup 5 --iters 100` | `./test_ag_gemm ... 1 5 100` |
| `test_gemm_rs.py ... --has_bias` | `./test_ag_gemm ... 1 5 10 0 1` |
| `test_gemm_rs.py ... --debug` | `./test_ag_gemm ... 1 5 10 0 0 1` |

## ✅ 实现的功能

- ✅ **性能测试**: 预热 + 多次迭代测量
- ✅ **多 GPU**: 支持多 GPU 并行（通过线程）
- ✅ **Bias 支持**: 可选的 bias 参数
- ✅ **权重转置**: 可选的权重转置
- ✅ **调试模式**: 简化数据用于验证
- ✅ **命令行接口**: 灵活的参数配置
- ✅ **文档完善**: 多个文档覆盖不同方面

## ❌ 移除的功能 (按需求)

- ❌ **Triton 实现**: 按用户需求移除
- ❌ **PyTorch 基线**: 简化为仅测试 Flux
- ❌ **多数据类型**: 当前仅支持 FP16
- ❌ **Profiling**: 未集成 torch.profiler 等效功能

## 🔧 依赖项

### 编译时依赖
- CUDA Toolkit
- Cutlass (3rdparty/cutlass)
- Flux 核心库 (flux_cuda)
- CMake 3.17+

### 运行时依赖
- 支持 P2P 的多 GPU 系统
- CUDA 兼容的 GPU (SM 7.0+)

## 📈 性能指标

测试输出以下指标：
- **GEMM 时间**: 矩阵乘法计算时间
- **通信时间**: AllGather 通信时间（在 AG+GEMM 中通常为 0，因为重叠）
- **总时间**: GEMM + 通信时间

## 🐛 故障排除

### 编译错误

1. **找不到 CUDA**
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   ```

2. **Cutlass 头文件缺失**
   - 确保子模块已初始化：`git submodule update --init --recursive`

3. **CMAKE 配置错误**
   - 清理 build 目录：`rm -rf build && mkdir build`

### 运行时错误

1. **GPU 数量不足**
   - 错误: "CUDA error: invalid device ordinal"
   - 解决: 确保 TP 参数 ≤ 可用 GPU 数量

2. **P2P 访问失败**
   - 检查: `nvidia-smi topo -m`
   - 确保 GPU 之间有 P2P 连接

3. **内存不足**
   - 减小矩阵维度 (M, N, K)
   - 减少并行度 (TP)

## 📚 相关文档

- **使用文档**: `README.md`
- **对比分析**: `COMPARISON.md`
- **Python 参考**: `/Users/min.yang/learncode/flux/test/python/gemm_rs/test_gemm_rs.py`
- **Flux 文档**: `/Users/min.yang/learncode/flux/docs/`

## 🎯 测试目标

本测试实现的主要目标：

1. ✅ **去除 Triton 依赖**: 提供纯 Flux C++ 实现
2. ✅ **性能基准**: 测量 AG+GEMM 操作的性能
3. ✅ **正确性验证**: 确保操作结果正确
4. ✅ **易用性**: 提供便捷的脚本和文档
5. ✅ **可扩展性**: 代码结构清晰，易于扩展

## 📞 反馈与改进

如需添加功能或报告问题，请参考：
- `README.md` 中的故障排除部分
- `COMPARISON.md` 中的扩展建议
- Flux 项目的 GitHub Issues

## 📄 许可证

所有文件遵循 Apache License 2.0，与 Flux 项目保持一致。

---

**创建日期**: 2025-11-28  
**参照版本**: test_gemm_rs.py  
**状态**: ✅ 完成并可用

