#!/bin/bash
# 编译 test_ag_gemm 的辅助脚本

set -e

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"

echo "项目根目录: $PROJECT_ROOT"

# 创建 build 目录（如果不存在）
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# 运行 CMake 配置
echo "正在配置 CMake..."
cmake .. -DBUILD_TEST=ON

# 编译 test_ag_gemm
echo "正在编译 test_ag_gemm..."
make test_ag_gemm -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "✅ 编译完成！"
echo "可执行文件位置: $BUILD_DIR/src/ag_gemm/test/test_ag_gemm"
echo ""
echo "运行示例:"
echo "  $BUILD_DIR/src/ag_gemm/test/test_ag_gemm 2048 10240 40960 8"

