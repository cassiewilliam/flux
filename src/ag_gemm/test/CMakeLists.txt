cmake_minimum_required(VERSION 3.17)

add_executable(test_ag_gemm_fp8 test_ag_gemm_fp8.cpp)
target_link_libraries(test_ag_gemm_fp8 PUBLIC flux_cuda)