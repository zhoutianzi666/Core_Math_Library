#pragma once
#include <stdio.h>

#include <iostream>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

#include <algorithm>

#include "cutlass/cutlass.h"

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/conv/kernel/default_depthwise_fprop.h"
#include "cutlass/conv/device/direct_convolution.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

void cutlass_nhwc_conv_depthwise(ConvAllParams params) {




constexpr int groups_per_cta = 16;
using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;
using FilterShape = cutlass::MatrixShape<3, 3>;

// 你看下面规约的维度总是FilterShape::kCount这个数字还是很小的！比一般的卷积小很多哦！
using ThreadblockShape =
    cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;
using WarpShape = cutlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;
using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
        1,
        ThreadBlockOutputShape::kN,
        ThreadBlockOutputShape::kH,
        ThreadBlockOutputShape::kW>;

// This code section describes the epilogue part of the kernel, we use default value
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    cutlass::half_t,               
    1,  // The number of elements per vectorized. memory access. This becomes the vector width of
                                 // math instructions in the epilogue too.
    cutlass::half_t,          // Data type of accumulator
    float,
    cutlass::epilogue::thread::ScaleType::Default>;  // Epilogue scaling operation.

using DepthwiseDirect2dConv = typename cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::layout::TensorNHWC,
    cutlass::half_t,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm70,
    ThreadblockShape,
    ThreadBlockOutputShape,
    FilterShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    SwizzleThreadBlock,
    4,
    cutlass::arch::OpMultiplyAdd,
    // This code section describe iterator algorithm selected is kFixedStrideDilation
    cutlass::conv::IteratorAlgorithm::kFixedStrideDilation,
    cutlass::conv::StrideSupport::kStrided,
    cutlass::MatrixShape<1, 1>,
    cutlass::MatrixShape<1, 1>>::Kernel;

  using Direct2dConv = cutlass::conv::device::DirectConvolution<DepthwiseDirect2dConv>;
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
  auto residual = params.residual;
  auto output = params.output;

  int groups = params.groups;
  int kc = ic / groups;

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, kc}, {pad_h0, 0, pad_w0, 0},
      {stride_h, stride_w}, {dilation_h, dilation_w}, {batch, oh, ow, oc}, 
      cutlass::conv::Mode::kCrossCorrelation,
      1, groups);

  typename Direct2dConv::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)weight, {kc, kc * kw, kc * kw * kh}},
      {(cutlass::half_t *)bias, {0, 0, 0}},
      {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f},
       {(cutlass::half_t *)residual, {kc, kc * kw, kc * kw * kh}},
       };
       // 这里residual被我用来当作临时空间了哦！
      // cutlass::conv::SplitKMode::kParallel 也可以用啊，但是啥时候会快呢？

  Direct2dConv implicit_gemm_op;
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




