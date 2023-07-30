project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)
# 3bf95e90c2a170bc56ee963517172217acad4345
include_directories("../../cutlass/include")


# nvcc flags
# cutlass 2.11 
set(CUDA_NVCC_FLAGS -g -gencode arch=compute_75,code=sm_75;)

CUDA_ADD_EXECUTABLE(a.out ${all_cu})
target_link_libraries(a.out cublas)
target_link_libraries(a.out cudnn)
