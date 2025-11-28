#!/bin/bash
# 运行 test_ag_gemm 的辅助脚本

set -e

# 获取项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
BUILD_DIR="$PROJECT_ROOT/build"

TEST_BINARY="$BUILD_DIR/src/ag_gemm/test/test_ag_gemm"

# 检查可执行文件是否存在
if [ ! -f "$TEST_BINARY" ]; then
    echo "❌ 错误: 找不到测试可执行文件"
    echo "请先运行: ./build_test.sh"
    exit 1
fi

# 默认参数（与 Python 版本对应）
M=${1:-2048}
N=${2:-10240}
K=${3:-40960}
TP=${4:-8}
NNODES=${5:-1}
WARMUP=${6:-5}
ITERS=${7:-10}
TRANSPOSE_WEIGHT=${8:-0}
HAS_BIAS=${9:-0}
DEBUG=${10:-0}

echo "=== 运行 AG+GEMM 测试 ==="
echo "参数: M=$M N=$N K=$K TP=$TP"
echo "NNodes=$NNODES Warmup=$WARMUP Iters=$ITERS"
echo "Transpose=$TRANSPOSE_WEIGHT Bias=$HAS_BIAS Debug=$DEBUG"
echo ""

# 运行测试
"$TEST_BINARY" $M $N $K $TP $NNODES $WARMUP $ITERS $TRANSPOSE_WEIGHT $HAS_BIAS $DEBUG

echo ""
echo "✅ 测试完成"

