project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;)
CUDA_ADD_EXECUTABLE(a.out ${all_cu})

target_link_libraries(a.out cublas)
target_link_libraries(a.out cudnn)



