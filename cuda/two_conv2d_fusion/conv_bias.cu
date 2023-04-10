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




void cutlass_nhwc_conv_bias(ConvAllParams p0, ConvAllParams p1) {


    using EpilogueOutputOp0 = 
    cutlass::epilogue::thread::LinearCombinationRelu<
    cutlass::half_t,
      4,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling
    >;
    
    using EpilogueOutputOp1 = 
    cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,
      8,
      cutlass::half_t,
      cutlass::half_t,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling
    >;
    
    using B2bConv2dFpropKernel = typename cutlass::conv::kernel::DefaultB2bConv2dFprop<
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t, cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::GemmShape<16, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    EpilogueOutputOp0,
    EpilogueOutputOp1,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kOptimized,
    false
    >::Kernel;
    
    using B2bConv2dFprop = cutlass::conv::device::B2bImplicitGemmConvolution<B2bConv2dFpropKernel>;    

    B2bConv2dFprop implicit_gemm_op;

  auto mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem0_size(
    {p0.batch, p0.ih, p0.iw, p0.ic}, 
    {p0.oc, p0.kh, p0.kw, p0.ic}, 
    {p0.pad_h0, 0, p0.pad_w0, 0},
    {p0.stride_h, p0.stride_w}, 
    {p0.dilation_h, p0.dilation_w}, 
    {p0.batch, p0.oh, p0.ow, p0.oc}, 
    mode, 1, 1);

cutlass::conv::Conv2dProblemSize problem1_size(
        {p1.batch, p1.ih, p1.iw, p1.ic}, 
        {p1.oc, p1.kh, p1.kw, p1.ic}, 
        {p1.pad_h0, 0, p1.pad_w0, 0},
        {p1.stride_h, p1.stride_w}, 
        {p1.dilation_h, p1.dilation_w}, 
        {p1.batch, p1.oh, p1.ow, p1.oc}, 
        mode, 1, 1);

  typename  B2bConv2dFprop::Arguments b2b_conv2d_args(
    problem0_size,
    problem1_size,
    {(cutlass::half_t *)p0.input, {p0.ic, p0.ic * p0.iw, p0.ic * p0.iw * p0.ih}},
    {(cutlass::half_t *)p0.weight, {p0.ic, p0.ic * p0.kw, p0.ic * p0.kw * p0.kh}},
    {(cutlass::half_t *)p0.bias, {0, 0, 0}},
    {(cutlass::half_t *)(nullptr), {p0.oc}},
    {(cutlass::half_t *)p0.bias, {p0.oc}},
    {(cutlass::half_t *)p1.weight, {p1.ic, p1.ic * p1.kw, p1.ic * p1.kw * p1.kh}},
    {(cutlass::half_t *)p1.bias, {0, 0, 0}},
    {(cutlass::half_t *)p1.output, {p1.oc, p1.oc * p1.ow, p1.oc * p1.ow * p1.oh}},
    {cutlass::half_t(1), cutlass::half_t(0)},
    {cutlass::half_t(1), cutlass::half_t(1)},
    cutlass::conv::SplitKMode::kSerial
  );


  cutlass::Status status = implicit_gemm_op.can_implement(b2b_conv2d_args);

  size_t bytes = implicit_gemm_op.get_workspace_size(b2b_conv2d_args);
  void *workspace;
  assert(bytes == 0);
 // cudaMalloc(&workspace, bytes);

  CUTLASS_CHECK(status);
  status = implicit_gemm_op.initialize(b2b_conv2d_args, workspace);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op();
  CUTLASS_CHECK(status);
 // cudaFree(workspace);

}
