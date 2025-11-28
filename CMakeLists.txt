cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(FLUX LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules/")

# cmake global settings
set(CMAKE_EXPORT_COMPILE_COMMANDS
  ON
  CACHE INTERNAL "")
set(BUILD_TEST
  ON
  CACHE INTERNAL "Build unit tests")
set(ENABLE_NVSHMEM
  ON
  CACHE INTERNAL "Use NVSHMEM to transfer data")
set(CUTLASS_TRACE
  OFF
  CACHE INTERNAL "Print CUTLASS Host Trace info")
set(FLUX_DEBUG
  OFF
  CACHE INTERNAL "Define FLUX_DEBUG")
option(WITH_PROTOBUF "build with protobuf" OFF)
option(WITH_TRITON_AOT "build with triton Ahead-of-Time" OFF)
message("NVShmem Support: ${ENABLE_NVSHMEM}")

# find cuda specify cuda path if other than default set(CUDA_TOOLKIT_ROOT_DIR
# /path/to/installed/cuda)
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")

if(CUDAToolkit_VERSION VERSION_LESS "11.0")
  message(FATAL_ERROR "requires cuda to be >= 11.0")
elseif(CUDAToolkit_VERSION VERSION_LESS "12.0")
  set(CUDAARCHS
    "80"
    CACHE STRING "CUDA Architectures")
elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.4")
  set(CUDAARCHS
    "80;89;90"
    CACHE STRING "CUDA Architectures")
else()
  set(CUDAARCHS
    "80;90"
    CACHE STRING "CUDA Architectures")
endif()

set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})
message(STATUS "CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

set(SM_CORES "")
option(GPU_SM_CORES "Manually specify GPU SM core number (e.g., '108;78'). Overrides auto-detection." "")

if(GPU_SM_CORES)
  set(SM_CORES ${GPU_SM_CORES})
  message(STATUS "Generator: Using manually specified GPU SM core number: ${SM_CORES}")
else()
  message(STATUS "Generator: GPU_SM_CORES not specified, attempting auto-detection by GPU name.")

  find_program(NVIDIA_SMI_EXECUTABLE nvidia-smi)

  if(NVIDIA_SMI_EXECUTABLE)
    execute_process(
      COMMAND ${NVIDIA_SMI_EXECUTABLE} --query-gpu=name --format=csv,noheader
      OUTPUT_VARIABLE DETECTED_GPU_NAMES
      RESULT_VARIABLE NVIDIA_SMI_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NVIDIA_SMI_RESULT EQUAL 0 AND DETECTED_GPU_NAMES)
      string(REPLACE "\n" ";" GPU_NAME_LIST ${DETECTED_GPU_NAMES})
      set(SM_CORE_LIST "")

      foreach(GPU_NAME ${GPU_NAME_LIST})
        set(CURRENT_SM_COUNT 0)

        if(GPU_NAME MATCHES "A100")
          set(CURRENT_SM_COUNT 108)
        elseif(GPU_NAME MATCHES "H800")
          set(CURRENT_SM_COUNT 132)
        elseif(GPU_NAME MATCHES "H100")
          set(CURRENT_SM_COUNT 132)
        elseif(GPU_NAME MATCHES "L20")
          set(CURRENT_SM_COUNT 92)
        elseif(GPU_NAME MATCHES "H20")
          set(CURRENT_SM_COUNT 78)
        endif()

        if(NOT CURRENT_SM_COUNT EQUAL 0)
          list(APPEND SM_CORE_LIST ${CURRENT_SM_COUNT})
        else()
          message(WARNING "Generator: Could not determine SM count for GPU '${GPU_NAME}'. Please add it to the lookup list in CMakeLists.txt.")
        endif()
      endforeach()

      if(SM_CORE_LIST)
        list(REMOVE_DUPLICATES SM_CORE_LIST)
        list(JOIN SM_CORE_LIST ";" SM_CORES)
        message(STATUS "Generator: Auto-detected SM cores from GPU names: ${SM_CORES}")
      endif()
    else()
      message(WARNING "Generator: nvidia-smi command failed or no GPUs detected. SM core filtering will be disabled.")
    endif()
  else()
    message(WARNING "Generator: nvidia-smi not found. SM core filtering will be disabled.")
  endif()
endif()

set(CUDA_ARCH_FLAGS)

foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND CUDA_ARCH_FLAGS
    "-gencode=arch=compute_${ARCH},code=\\\"sm_${ARCH},compute_${ARCH}\\\"")
endforeach()

string(JOIN " " JOINED_CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${JOINED_CUDA_ARCH_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wno-psabi")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -DCUTLASS_DEBUG_TRACE_LEVEL=0")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

if(CUTLASS_TRACE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUTLASS_DEBUG_TRACE_LEVEL=1")
endif()

if(FLUX_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFLUX_DEBUG")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFLUX_DEBUG")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# force use sm90a for cutlass
string(REGEX REPLACE "sm_90" "sm_90a" CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS})
string(REGEX REPLACE "compute_90" "compute_90a" CMAKE_CUDA_FLAGS
  ${CMAKE_CUDA_FLAGS})

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/library/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/profiler/include)

set(COMMON_LIB_DIRS "")
list(APPEND COMMON_LIB_DIRS "${CUDAToolkit_LIBRARY_DIR}")
message(ENABLE_NVSHMEM "ENABLE_NVSHMEM is set to: ${ENABLE_NVSHMEM}")

if(ENABLE_NVSHMEM)
  add_definitions(-DFLUX_SHM_USE_NVSHMEM)
  message(STATUS "NVSHMEM home dir: ${NVSHMEM_HOME}")

  if(NOT EXISTS ${NVSHMEM_HOME})
    message(
      FATAL_ERROR "NVSHMEM not found. Please run `pip3 install nvidia-nvshmem-cu12` first.")
  endif()

  list(APPEND COMMON_HEADER_DIRS "${NVSHMEM_HOME}/include")
  list(APPEND COMMON_LIB_DIRS "${NVSHMEM_HOME}/lib")
endif()

# append headers explicitly for .cu files, in order to enable vscode clangd
# intellisense
foreach(inc_dir ${COMMON_HEADER_DIRS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -I${inc_dir}")
endforeach()

message("final CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

include_directories(${COMMON_HEADER_DIRS})

link_directories(${COMMON_LIB_DIRS})

if(WITH_PROTOBUF)
  find_package(Protobuf REQUIRED)
  add_subdirectory(proto)
endif()

add_subdirectory(src)