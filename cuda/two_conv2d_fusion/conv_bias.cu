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
#include "device/b2b_implicit_gemm_convolution.h"

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;

using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
    cutlass::half_t,  // Data type of output matrix.
    8,
    float,   // Data type of accumulator
    float>;  // Data type for alpha/beta in linear combination

    using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
      4,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;

  using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
      8,
      float,
      float,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;


using B2bConv2dFpropKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::GemmShape<16, 128, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    false
  >::Kernel;


using B2bConv2dFprop = cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;


using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t,
    cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
    float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2,
    cutlass::arch::OpMultiplyAdd, IteratorAlgorithm,
    cutlass::conv::StrideSupport::kUnity, 8, 8>::Kernel;

using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

void cutlass_nhwc_conv_bias(ConvAllParams p0, ConvAllParams p1) {
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;


  cutlass::conv::Conv2dProblemSize problem0_size(
    {p0.batch, p0.ih, p0.iw, p0.ic}, 
    {p0.oc, p0.kh, p0.kw, p0.ic}, 
    {p0.pad_h0, p0.pad_h1, p0.pad_w0, p0.pad_w1},
    {p0.stride_h, p0.stride_w}, {p0.dilation_h, p0.dilation_w}, 
    {p0.batch, p0.oh, p0.ow, p0.oc}, 
    mode, 1, 1);

cutlass::conv::Conv2dProblemSize problem1_size(
        {p1.batch, p1.ih, p1.iw, p1.ic}, {p1.oc, p1.kh, p1.kw, p1.ic}, 
        {p1.pad_h0, p1.pad_h1, p1.pad_w0, p1.pad_w1},
        {p1.stride_h, p1.stride_w}, {p1.dilation_h, p1.dilation_w}, 
        {p1.batch, p1.oh, p1.ow, p1.oc}, 
        mode, 1, 1);

  typename  B2bConv2dFprop::Arguments b2b_conv2d_args(
    problem0_size,
    problem1_size,
    tensor_A0.device_ref(),
    {(cutlass::half_t *)p0.weight, {p0.ic, p0.ic * p0.kw, p0.ic * p0.kw * p0.kh}},
    tensor_C0.device_ref(),
    tensor_Scale0.device_ref(),
    tensor_Bias0.device_ref(),
    {(cutlass::half_t *)p1.weight, {p1.ic, p1.ic * p1.kw, p1.ic * p1.kw * p1.kh}},
    {tensor_Bias1.device_data(), typename B2bConv2d::LayoutC::Stride(0)},
    tensor_D1_computed.device_ref(),
    {alpha0, beta0},
    {alpha1, beta1},
    split_k_mode
  );


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

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h0, pad_h1, pad_w0, pad_w1},
      {stride_h, stride_w}, {dilation_h, dilation_w}, {batch, oh, ow, oc}, mode,
      1, 1);

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
  cudaMalloc(&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op();
  CUTLASS_CHECK(status);
  cudaFree(workspace);

}
