project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
find_package(CUDA)
include_directories("/zhoukangkang/2022-04-28inference_try/cutlass/include")
include_directories("/zhoukangkang/2022-04-28inference_try/cutlass/tools/util/include/")
include_directories("/zhoukangkang/2022-04-28inference_try/cutlass/tools/util/include/")
include_directories("/zhoukangkang/2022-04-28inference_try/cutlass/examples/13_two_tensor_op_fusion/")

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_75,code=sm_75;)
CUDA_ADD_EXECUTABLE(a.out ${all_cu})
target_link_libraries(a.out cublas)
target_link_libraries(a.out cudnn)
