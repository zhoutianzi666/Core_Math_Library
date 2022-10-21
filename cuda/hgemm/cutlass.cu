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

using ElementAccumulator = float;  
using ElementComputeEpilogue = ElementAccumulator;
using ElementInputA = cutlass::half_t;
using ElementInputB = cutlass::half_t;
using ElementOutput = cutlass::half_t;

// Note that if the output is column major, the bias has to be per row. i.e.
// every row has different bias. If the output is row major, the bias has to be
// per column, i.e. every column has different bias. Below list some other
// notices:
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the
//      row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

using LayoutInputA = cutlass::layout::ColumnMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::ColumnMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm75;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 64, 32>;
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 8, N = 8, K = 4

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value,  // <- this is the number of elements per
                                      // vectorized memory access. For half
                                      // precision, it's 8 elements. This
                                      // becomes the vector width of math
                                      // instructions in epilogue too
    ElementAccumulator,               // <- data type of accumulator
    ElementComputeEpilogue>;          // <- data type for alpha/beta in linear
                                      // combination function
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
    ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages, 8, 8 ,false>;

cudaError_t CutlassHgemmNN(int M, int N, int K, DATATYPE alpha,
                           DATATYPE const *A, int lda, DATATYPE const *B,
                           int ldb, DATATYPE beta, DATATYPE *C, int ldc) {
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments
  // to launch instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{
      problem_size,               // <- problem size of matrix multiplication
      {(ElementInputA *)A, lda},  // <- reference to matrix A on device
      {(ElementInputB *)B, ldb},  // <- reference to matrix B on device
      {(ElementOutput *)C, ldc},
      {(ElementOutput *)C, ldc},  // <- reference to matrix D on device
      {alpha},                    // <- alpha
      split_k_slices};            // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix
  // multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  size_t bytes = Gemm::get_workspace_size(arguments);
  void * workspace;
  cudaMalloc((void**)&workspace, bytes);

  Gemm gemm_op;

  cutlass::Status status = gemm_op.can_implement(arguments);
  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace);

  // Launch initialized CUTLASS kernel
  status = gemm_op();
  return cudaSuccess;
}
