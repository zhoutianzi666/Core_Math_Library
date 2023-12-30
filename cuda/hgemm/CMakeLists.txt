project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)

#  git checkout v2.11.0
include_directories("../../cutlass/include")
include_directories("../../cutlass/tools/util/include")


# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75 -std=c++17 --ptxas-options --verbose;)
CUDA_ADD_EXECUTABLE(a.out ${all_cu})
target_link_libraries(a.out cublas)
target_link_libraries(a.out cudnn)
