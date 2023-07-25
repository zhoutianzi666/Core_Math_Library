project (a.out)
cmake_minimum_required(VERSION 3.1...3.21 FATAL_ERROR)

file(GLOB all_cu "*.cu")
list(APPEND all_cu "./cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.cu")
find_package(CUDA)

# cc85b64
include_directories("../cutlass/include")
include_directories("../cutlass/tools/util/include")


include_directories("./")
include_directories("./cutlass_kernels")
include_directories("./cutlass_kernels/fpA_intB_gemm")

# nvcc flags
set(CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70;)
CUDA_ADD_EXECUTABLE(a.out ${all_cu})
#target_link_libraries(a.out cublas)
#target_link_libraries(a.out cudnn)
