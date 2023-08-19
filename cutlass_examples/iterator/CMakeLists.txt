project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

#file(GLOB all_cu "*.cu")
find_package(CUDA)
include_directories("../../cutlass/include")

set(CUDA_NVCC_FLAGS -std=c++17;)
#CUDA_ADD_EXECUTABLE(a.out ${all_cu})
CUDA_ADD_EXECUTABLE(a.out PitchLinearWarpStripedThreadMap.cu)
target_link_libraries(a.out cublas)
