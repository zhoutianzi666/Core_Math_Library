
#pragma once
#include <stdio.h>
#include <vector>
#include <functional>
#include <iostream>
#include <mutex>
#include <map>
#include "utility.h"

using DATATYPE = half;

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"

void check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    printf("CUTLASS 不能实施\n");
  }
}

# define kernel_instantiate(num, T0, T1, T2, W0, W1, W2, I0, I1, I2) \
cutlass::Status cutlass_nhwc_conv_relu_##num(ConvAllParams params) { \
using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop< \
    cutlass::half_t, cutlass::layout::TensorNHWC, \
    cutlass::half_t, cutlass::layout::TensorNHWC, \
    cutlass::half_t, cutlass::layout::TensorNHWC,\
    float, \
    cutlass::arch::OpClassTensorOp, \
    cutlass::arch::Sm75,\
    cutlass::gemm::GemmShape<T0, T1, T2>, \
    cutlass::gemm::GemmShape<W0, W1, W2>,\
    cutlass::gemm::GemmShape<I0, I1, I2>, \
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float, float>,\
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>, 2,\
    cutlass::arch::OpMultiplyAdd, \
    cutlass::conv::IteratorAlgorithm::kOptimized,\
    cutlass::conv::StrideSupport::kStrided, 8, 8>::Kernel;\
using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;\
  int batch = params.batch;\
  int ih = params.ih;\
  int iw = params.iw;\
  int ic = params.ic;\
  int oc = params.oc;\
  int kh = params.kh;\
  int kw = params.kw;\
  int pad_h0 = params.pad_h0;\
  int pad_h1 = params.pad_h1;\
  int pad_w0 = params.pad_w0;\
  int pad_w1 = params.pad_w1;\
  int stride_h = params.stride_h;\
  int stride_w = params.stride_w;\
  int dilation_h = params.dilation_h;\
  int dilation_w = params.dilation_w;\
  int oh = params.oh;\
  int ow = params.ow;\
  auto input = params.input;\
  auto weight = params.weight;\
  auto bias = params.bias;\
  auto output = params.output;\
  int groups = params.groups;\
  cutlass::conv::Conv2dProblemSize problem_size(\
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h0, 0, pad_w0, 0},\
      {stride_h, stride_w}, \
      {dilation_h, dilation_w},\ 
      {batch, oh, ow, oc}, \
      cutlass::conv::Mode::kCrossCorrelation,\
      1, groups);\
  typename ImplicitGemm::Arguments arguments{\
      problem_size,\
      {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},\
      {(cutlass::half_t *)weight, {ic, ic * kw, ic * kw * kh}},\
      {(cutlass::half_t *)bias, {0, 0, 0}},\
      {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},\
      {1.f, 1.f},\
      cutlass::conv::SplitKMode::kSerial};\
      /* cutlass::conv::SplitKMode::kParallel 也可以用啊，但是啥时候会快呢？*/\
  ImplicitGemm implicit_gemm_op;\
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);\
  void *workspace;\
  cudaMalloc((void **)&workspace, bytes);\
  cutlass::Status status = implicit_gemm_op.can_implement(arguments);\
  check(status);\
  status = implicit_gemm_op.initialize(arguments, workspace);\
  check(status);\
  status = implicit_gemm_op();\
  check(status);\
  return cutlass::Status::kSuccess;\
}



kernel_instantiate(0, 64, 64, 64, 32, 32, 64, 16, 8, 8)
kernel_instantiate(1, 64, 32, 64, 32, 32, 64, 16, 8, 8)
kernel_instantiate(2, 128, 32, 64, 32, 32, 64, 16, 8, 8)
kernel_instantiate(3, 128, 64, 64, 32, 32, 64, 16, 8, 8)
kernel_instantiate(4, 64, 64, 32, 32, 32, 32, 16, 8, 8)
kernel_instantiate(5, 64, 128, 32, 32, 64, 32, 16, 8, 8)
kernel_instantiate(6, 64, 128, 64, 64, 64, 32, 16, 8, 8)
kernel_instantiate(7, 64, 256, 32, 64, 64, 32, 16, 8, 8)
kernel_instantiate(8, 128, 64, 32, 64, 32, 32, 16, 8, 8)
// 下面这俩个配置是我自己加的！
kernel_instantiate(9, 16, 32, 16, 16, 32, 16, 16, 8, 8)
kernel_instantiate(10, 128, 64, 64, 32, 64, 64, 16, 8, 8)

std::vector<std::function<cutlass::Status(const ConvAllParams)>>
    cutlass_nhwc_conv_relu_all_func =  {
    cutlass_nhwc_conv_relu_0,
    cutlass_nhwc_conv_relu_1,
    cutlass_nhwc_conv_relu_2,
    cutlass_nhwc_conv_relu_3,
    cutlass_nhwc_conv_relu_4,
    cutlass_nhwc_conv_relu_5,
    cutlass_nhwc_conv_relu_6,
    cutlass_nhwc_conv_relu_7,
    cutlass_nhwc_conv_relu_8,
    cutlass_nhwc_conv_relu_9,
    cutlass_nhwc_conv_relu_10,
};


std::map<std::vector<int>, int> map_problem_cutlass_nhwc_conv_relu;
std::mutex cutlass_nhwc_conv_relu_mutex;


void cutlass_nhwc_conv_relu(ConvAllParams params) {
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  //int pad_h0 = params.pad_h0;
  //int pad_w0 = params.pad_w0;
  int groups = params.groups;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;

  std::vector<int> problem_size = {
      batch, ic, ih, iw, kh, kw, oc, groups, stride_h, stride_w};

  if (map_problem_cutlass_nhwc_conv_relu.count(problem_size)) {
    cutlass_nhwc_conv_relu_all_func[map_problem_cutlass_nhwc_conv_relu.at(problem_size)](
        params);
    return;
  }

  int best_config_index = ProfileToGetBestConfig(
      cutlass_nhwc_conv_relu_all_func, params);

  std::lock_guard<std::mutex> guard(cutlass_nhwc_conv_relu_mutex);

  map_problem_cutlass_nhwc_conv_relu[problem_size] = best_config_index;
  cutlass_nhwc_conv_relu_all_func[best_config_index](params);
}