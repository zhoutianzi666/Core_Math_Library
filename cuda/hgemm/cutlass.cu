#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
cudaError_t CutlassHgemmNN1(int M, int N, int K, DATATYPE alpha,
                            DATATYPE const *A, int lda, DATATYPE const *B,
                            int ldb, DATATYPE beta, DATATYPE *C, int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions
  // including the following example for single-precision GEMM. Typical values
  // are used as
  // default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<DATATYPE,      // Data-type of A matrix
                                  ColumnMajor,   // Layout of A matrix
                                  DATATYPE,      // Data-type of B matrix
                                  ColumnMajor,   // Layout of B matrix
                                  DATATYPE,      // Data-type of C matrix
                                  ColumnMajor>;  // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible
  // in host code and passed to kernels by value. These may include pointers,
  // strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel
  // entry.
  //
  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},   // Tensor-ref for source matrix A
                              {B, ldb},   // Tensor-ref for source matrix B
                              {C, ldc},   // Tensor-ref for source matrix C
                              {C, ldc},   // Tensor-ref for destination matrix D
                              // (may be different memory than source
                              // C matrix)
                              {alpha, beta});  // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

#include <algorithm>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t, 
    128 / cutlass::sizeof_bits<cutlass::half_t>::value,
    float,               
    float>;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    cutlass::half_t,
    cutlass::layout::ColumnMajor,
    float, 
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,    
    EpilogueOp, 
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2, 
    8, 8 ,
    true>;

using Gemm_80 = cutlass::gemm::device::Gemm<
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  cutlass::half_t,
  cutlass::layout::ColumnMajor,
  float, 
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<128, 128, 32>,
  cutlass::gemm::GemmShape<64, 64, 32>,
  cutlass::gemm::GemmShape<16, 8, 16>,    
  EpilogueOp, 
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
  2, 
  8, 8 ,
  true>;


cudaError_t CutlassHgemmNN(int M, int N, int K, DATATYPE alpha,
                           DATATYPE const *A, int lda, DATATYPE const *B,
                           int ldb, DATATYPE beta, DATATYPE *C, int ldc) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);
  // Split K dimension into 1 partitions
  int split_k_slices = 2;
  typename Gemm::Arguments arguments{
      problem_size,               
      {(cutlass::half_t *)A, lda},  
      {(cutlass::half_t *)B, ldb},  
      {(cutlass::half_t *)C, ldc},
      {(cutlass::half_t *)C, ldc},  
      {alpha},                    
      split_k_slices};   

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  size_t bytes = Gemm::get_workspace_size(arguments);
  printf("需要临时空间为: %d 字节\n", bytes);
  if(split_k_slices == 1) {
    assert(bytes == 0);
  }

  void * workspace;
  cudaMalloc((void**)&workspace, bytes);

  Gemm gemm_op;
  cutlass::Status status = gemm_op.can_implement(arguments);
  status = gemm_op.initialize(arguments, workspace);


  // Launch initialized CUTLASS kernel
  status = gemm_op();
  return cudaSuccess;
}
