#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

using DATATYPE = float;

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
using ElementComputeEpilogue = float;  // Data type of epilogue computation (alpha, beta)
using ElementInputA = float;  // Data type of elements in input tensor
using ElementInputB = float;  // Data type of elements in input tensor
using ElementOutput =  float;  // Data type of elements in output tensor
using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;
using MMAOp = cutlass::arch::OpClassSimt;
using SmArch = cutlass::arch::Sm75;
using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 8>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
constexpr int NumStages = 2;
static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,  // Data type of output matrix.
    1,
    ElementAccumulator,       // Data type of accumulator
    ElementComputeEpilogue,
    cutlass::epilogue::thread::ScaleType::Nothing>;  // Data type for alpha/beta in linear combination

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
    LayoutOutput, ElementAccumulator, MMAOp, SmArch, ThreadblockShape,
    WarpShape, InstructionShape, EpilogueOp, SwizzleThreadBlock, NumStages,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_nhwc_conv1(const float *input, const float *weight, float *output,
                       int batch, int ic, int ih, int iw, int kh, int kw,
                       int oc, int pad_h, int pad_w, int stride_h, int stride_w,
                       int oh, int ow, cudaStream_t stream) {
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h, pad_w, pad_h, pad_w},
      {stride_h, stride_w}, {stride_h, stride_w}, {batch, oh, ow, oc}, mode, 1);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(float*)input, {ic, ic * iw, ic * iw * ih}},
      {(float*)weight, {ic, ic * kw, ic * kw * kh}},
      {(float*)output, {oc, oc * ow, oc * ow * oh}},
      {(float*)output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 0.f}};

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
  void *workspace;
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op(stream);
  check(status);
}

// 下面是另一种方法，是从cutlass中拷贝来的！

#include "cutlass/cutlass.h"


  // Conv2dFprop Optimized kernel instance "cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1"
  using cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1 = 
  typename cutlass::conv::kernel::DefaultConv2dFprop<
    float, 
    cutlass::layout::TensorNHWC,
    float, 
    cutlass::layout::TensorNHWC,
    float, 
    cutlass::layout::TensorNHWC,
    float,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<128, 32, 8>,
    cutlass::gemm::GemmShape<32, 32, 8 >,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<
      float,
      1,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::Nothing
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, // cutlass::gemm::threadblock::GemmSplitKIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    cutlass::conv::StrideSupport::kStrided,
    1,
    1
  >::Kernel;


  using Operation_cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1 = cutlass::conv::device::ImplicitGemmConvolution<
    cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1>;

void cutlass_nhwc_conv(const float *input, const float *weight, float *output,
    int batch, int ic, int ih, int iw, int kh, int kw,
    int oc, int pad_h, int pad_w, int stride_h, int stride_w,
    int oh, int ow, cudaStream_t stream) {
cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

cutlass::conv::Conv2dProblemSize problem_size(
{batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h, pad_w, pad_h, pad_w},
{stride_h, stride_w}, {stride_h, stride_w}, {batch, oh, ow, oc}, mode, 1);

typename Operation_cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1::Arguments arguments{
problem_size,
{(float*)input, {ic, ic * iw, ic * iw * ih}},
{(float*)weight, {ic, ic * kw, ic * kw * kh}},
{(float*)output, {oc, oc * ow, oc * ow * oh}},
{(float*)output, {oc, oc * ow, oc * ow * oh}},
{1.f, 0.f}};

Operation_cutlass_simt_sfprop_optimized_128x128_8x2_nhwc_align1 implicit_gemm_op;
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
