#include <algorithm>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"


// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = C_DATATYPE;  // <- data type of epilogue operations
using ElementInputA = DATATYPE;                       // <- data type of elements in input matrix A
using ElementInputB = DATATYPE;                       // <- data type of elements in input matrix B
using ElementOutput = C_DATATYPE;                      // <- data type of elements in output matrix D

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<256, 128, 64>;
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>; 
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>; 

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                   
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;

cudaError_t CutlassIgemmNN_sm80(int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B,int ldb, 
                           float const *bias,
                           C_DATATYPE *C, int ldc) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // C = alpha * AB + beta * bias
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(1);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  // 下面 BROADCAST_DATATYPE 和 C_DATATYPE我看估计必须等相等的！
  // 必须的啊，因为这个矩阵乘法做的操作是 C = alpha * AB + beta * D
  // 这个C和D的数据类型可以不一样吗？
  typename Gemm::Arguments arguments{
      problem_size,          
       {(DATATYPE *)A, lda},  
       {(DATATYPE *)B, ldb},       
      {(BROADCAST_DATATYPE *)bias, 0},
      {(C_DATATYPE *)C, ldc},  
      {alpha, beta},                    
      split_k_slices};   

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  size_t bytes = Gemm::get_workspace_size(arguments);
  void * workspace;
  cudaMalloc((void**)&workspace, bytes);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  status = gemm_op.initialize(arguments, workspace);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  return cudaSuccess;
}
