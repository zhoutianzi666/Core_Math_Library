project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)

# 3bf95e90c2a170bc56ee963517172217acad4345
include_directories("../../cutlass/include")

set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;)
CUDA_ADD_EXECUTABLE(a.out ${all_cu})
target_link_libraries(a.out cublas)
