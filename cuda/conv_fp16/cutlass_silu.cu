#pragma once
#include <stdio.h>

#include <iostream>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

using DATATYPE = half;

#include <algorithm>

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kFewChannels;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
    cutlass::half_t,  // Data type of output matrix.
    1,
    float,   // Data type of accumulator
    float>;  // Data type for alpha/beta in linear combination

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t,
    cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<16, 32, 16>, cutlass::gemm::GemmShape<16, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm,
    cutlass::conv::StrideSupport::kUnity, 8, 8>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_nhwc_conv_bias_swish(const half *input, const half *weight,
                                  const half *bias, half *output, int batch,
                                  int ic, int ih, int iw, int kh, int kw,
                                  int oc, int pad_h, int pad_w, int stride_h,
                                  int stride_w, int oh, int ow) {
  auto check = [](cutlass::Status status) {
    if (status != cutlass::Status::kSuccess) {
      printf("不能实施\n");
    }
  };

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
      {1.f, 1.f},
      cutlass::conv::SplitKMode::kParallel};

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
