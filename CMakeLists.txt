# cmake version
cmake_minimum_required(VERSION 3.16.3)

# project info
project(CUDAops LANGUAGES CXX CUDA)

# set executable output path
set(PATH_EXECUTABLE bin)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_EXECUTABLE})
SET(EXECUTABLE_OUTPUT_PATH ../${PATH_EXECUTABLE})

# set library output path
set(PATH_LIBRARY lib)
execute_process( COMMAND ${CMAKE_COMMAND} -E make_directory ../${PATH_LIBRARY})
SET(LIBRARY_OUTPUT_PATH ../${PATH_LIBRARY})

# find CUDA Toolkit Extension library
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.7)
find_library(CUDA_NVTX_LIBRARY
  NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
  PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
  PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
  DOC "Location of the CUDA Toolkit Extension (NVTX) library"
  NO_DEFAULT_PATH
)

# ====================== OPERATOR LIBRARY ======================
file(GLOB_RECURSE SRC_OPERATORS "kernels/*.cu" "kernels/*.cpp" "kernels/**/*.cu" "kernels/**/*.cpp")
add_library(_operators ${SRC_OPERATORS})
target_compile_features(_operators PUBLIC cxx_std_11)
target_include_directories(_operators PUBLIC include)
target_compile_options(
  _operators PUBLIC 
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
)

# Don't do following operation!
# set_target_properties( 
#   _operators                   
#   PROPERTIES CUDA_SEPARABLE_COMPILATION ON
# )

target_include_directories(_operators PUBLIC include)

# ====================== src PROGRAM ======================
# >>> GNN aggregation
SET(SRC_GNN_AGGREGATION src/GNNaggregation)
file(GLOB_RECURSE SRC_GNN_AGGREGATION_CPP ${SRC_GNN_AGGREGATION}/*.cpp)
add_executable(basic_aggregation ${SRC_GNN_AGGREGATION}/basic_aggregation.cu ${SRC_GNN_AGGREGATION_CPP})
add_executable(tiled_aggregation ${SRC_GNN_AGGREGATION}/tiled_aggregation.cu ${SRC_GNN_AGGREGATION_CPP})
# >>> CSR SpMV
SET(SRC_CSR_SPMV src/SpMV)
file(GLOB_RECURSE SRC_CSR_SPMV_CPP ${SRC_CSR_SPMV}/*.cpp)
add_executable(csr_spmv_scalar ${SRC_CSR_SPMV}/csr_spmv_scalar.cu ${SRC_CSR_SPMV_CPP})


# >>> global configuration
set(PROFILING_TARGETS 
  basic_aggregation
  tiled_aggregation
  csr_spmv_scalar
)
foreach( profiling_target ${PROFILING_TARGETS} )
  target_link_libraries(${profiling_target} _operators nvToolsExt -lcublas -lcurand)
  target_compile_features(${profiling_target} PUBLIC cxx_std_11)
  target_include_directories(${profiling_target} PUBLIC include)
  target_compile_options(
    ${profiling_target} PRIVATE 
    # 编译器前后端: arch -> IR, code -> 后端 binary (针对硬件的二进制，小版本号兼容 sm_80 -> sm_86)
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
  )
endforeach( profiling_target ${PROFILING_TARGETS} )