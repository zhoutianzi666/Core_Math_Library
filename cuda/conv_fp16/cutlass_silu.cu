#pragma once
#include <stdio.h>

#include <iostream>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

#include <algorithm>

#include "cutlass/cutlass.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
    cutlass::half_t,  // Data type of output matrix.
    8,
    float,   // Data type of accumulator
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // Data type for alpha/beta in linear combination
// 因为是直接加上的bias，bias不需要beta放缩，所以我上面就直接用NoBetaScaling了。
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t,
    cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<16, 32, 16>, cutlass::gemm::GemmShape<16, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm,
    cutlass::conv::StrideSupport::kUnity, 1, 1>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_nhwc_conv_bias_swish(ConvAllParams params) {
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

  int batch = params.batch;
  int ih = params.ih;
  int iw = params.iw;
  int ic = params.ic;
  int oc = params.oc;
  int kh = params.kh;
  int kw = params.kw;
  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

  int oh = params.oh;
  int ow = params.ow;
  auto input = params.input;
  auto weight = params.weight;
  auto bias = params.bias;
  auto output = params.output;

  int groups = 1;

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h0, pad_h1, pad_w0, pad_w1},
      {stride_h, stride_w}, {dilation_h, dilation_w}, {batch, oh, ow, oc}, mode,
      1, groups);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)weight, {ic, ic * kw, ic * kw * kh}},
      {(cutlass::half_t *)bias, {0, 0, 0}},
      {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f},
      cutlass::conv::SplitKMode::kSerial};
      // cutlass::conv::SplitKMode::kParallel 也可以用啊，但是啥时候会快呢？

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
  // 其实bytes基本都是0，因为是kSerial。
  assert(bytes == 0);
  void *workspace;
  //cudaMalloc(&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op();
  CUTLASS_CHECK(status);
  // cudaFree还蛮费时间的！
  //cudaFree(&workspace, bytes);
}
