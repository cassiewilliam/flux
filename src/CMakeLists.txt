cmake_minimum_required(VERSION 3.17)

set(GEMM_OP_REG_INP_FILE ${CMAKE_CURRENT_SOURCE_DIR}/cuda/gemm_op_registry.cu.in)

# append 'flux/src/' to include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# create an object library for each op subfolder
function(flux_add_op_cu_obj_lib TARGET FILES)
  add_library(${TARGET} OBJECT ${FILES})
  set_target_properties(${TARGET} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(${TARGET} PUBLIC CUDA::cudart ${CMAKE_THREAD_LIBS_INIT})
  set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
endfunction()

set_property(GLOBAL PROPERTY FLUX_THS_TARGETS)

function(flux_add_ths_op_target TARGET)
  get_property(FLUX_THS_TARGETS_LOCAL GLOBAL PROPERTY FLUX_THS_TARGETS)
  set_property(GLOBAL PROPERTY FLUX_THS_TARGETS "${TARGET};${FLUX_THS_TARGETS_LOCAL}")
endfunction()

set(FLUX_CUDA_OP_TARGETS "") # cu op targets of subdirectory

include(ExternalProject)
ExternalProject_Add(
  FLUX_GENERATOR
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generator
  CONFIGURE_COMMAND ""
  BUILD_COMMAND "" # Prevents automatic build during build phase
  INSTALL_COMMAND "" # Prevents automatic install
)
ExternalProject_Get_property(FLUX_GENERATOR SOURCE_DIR)
ExternalProject_Get_property(FLUX_GENERATOR BINARY_DIR)
set(FLUX_GENERATOR_BINARY_DIR ${BINARY_DIR})
set(FLUX_GENERATOR_SOURCE_DIR ${SOURCE_DIR})

execute_process(
  COMMAND ${CMAKE_COMMAND} ${FLUX_GENERATOR_SOURCE_DIR}
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND ${CMAKE_COMMAND} --build . --parallel 8
  WORKING_DIRECTORY ${FLUX_GENERATOR_BINARY_DIR}
  COMMAND_ERROR_IS_FATAL ANY
)

add_subdirectory(ag_gemm)

# add_subdirectory(gemm_rs)
find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.3")
  add_subdirectory(quantization)
endif()

# if(ENABLE_NVSHMEM)
# # MoE OPs will not be open-sourced, so remain MoE OPs to be dependent on NVSHMEM
# add_subdirectory(moe_ag_scatter)
# add_subdirectory(moe_gather_rs)
# endif()
message(STATUS "op_targets: ${FLUX_CUDA_OP_TARGETS}")

# this must be after all op subdirectory added
add_subdirectory(cuda)
