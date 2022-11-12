#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

using DATATYPE = half;

#include <algorithm>

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"

void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    printf("不能实施\n");
  }
}

using ElementAccumulator = float;  // Data type of accumulator
using ElementComputeEpilogue =
    float;  // Data type of epilogue computation (alpha, beta)
using ElementInputA = cutlass::half_t;  // Data type of elements in input tensor
using ElementInputB = cutlass::half_t;  // Data type of elements in input tensor
using ElementOutput =
    cutlass::half_t;  // Data type of elements in output tensor
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;
using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm75;
using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 64>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
constexpr int NumStages = 2;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,  // Data type of output matrix.
    128 / cutlass::sizeof_bits<ElementOutput>::
              value,
    ElementAccumulator,       // Data type of accumulator
    ElementComputeEpilogue>;  // Data type for alpha/beta in linear combination

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ThreadblockShape,
    WarpShape, InstructionShape, EpilogueOp, SwizzleThreadBlock, NumStages,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm,
    cutlass::conv::StrideSupport::kStrided,
    8,
    8
    >::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_nhwc_conv(const half *input, const half *weight, const half *bias,
                       half *output,
                       int batch, int ic, int ih, int iw, int kh, int kw,
                       int oc, int pad_h, int pad_w, int stride_h, int stride_w,
                       int oh, int ow) {
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h, pad_w, pad_h, pad_w},
      {stride_h, stride_w}, {1, 1}, {batch, oh, ow, oc}, mode, 1);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)weight, {ic, ic * kw, ic * kw * kh}},
      {(cutlass::half_t *)bias, {0, 0, 0}},
      {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f}};

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
  void *workspace;
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
}
