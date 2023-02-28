
#pragma once
#include <stdio.h>
#include <cuda_fp16.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_leaky_relu.h"

#define WARMUP 10
#define REPEATE 100

using DATATYPE = half;
using C_DATATYPE = half;

void init(half *a, int size) {
    for (int i = 0; i < size; i++) {
      a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
    }
}

void cutlass_check(cutlass::Status status) {
  if (status != cutlass::Status::kSuccess) {
    printf("can not implement this conv\n");
  }
}

    using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultConv2dFprop<
      cutlass::half_t,
      cutlass::layout::TensorNHWC,
      cutlass::half_t,
      cutlass::layout::TensorNHWC,
      cutlass::half_t,
      cutlass::layout::TensorNHWC,
      float,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm75,
      cutlass::gemm::GemmShape<64, 64, 64>,
      cutlass::gemm::GemmShape<32, 32, 64>,
      cutlass::gemm::GemmShape<16,8,8>,
      cutlass::epilogue::thread::LinearCombinationSilu< cutlass::half_t, 8, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
      2,
      cutlass::arch::OpMultiplyAdd,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      8,
      8
    >::Kernel;


using ImplicitGemm =
    cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

int main(void) {
  int batch = 1;
  int ic = 32;
  int ih = 320;
  int iw = 320;
  int pad_h0 = 1;
  int pad_h1 = 1;
  int pad_w0 = 1;
  int pad_w1 = 1;
  int oc = 64;
  int kh = 3;
  int kw = 3;
  int stride_h = 2;
  int stride_w = 2;
  int dilation_h = 1;
  int dilation_w = 1;

  int oh = (ih + pad_h0 + pad_h1 - dilation_h * (kh - 1) - 1) / stride_h + 1;
  int ow = (iw + pad_w0 + pad_w1 - dilation_w * (kw - 1) - 1) / stride_w + 1;

  // Note input and weight is in CPU place
  DATATYPE *input, *weight, *bias;
  int input_size = batch * ic * ih * iw;
  int weight_size = oc * ic * kh * kw;
  int output_size = batch * oc * oh * ow;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  status = cudaMallocHost(&weight, sizeof(DATATYPE) * weight_size);
  status = cudaMallocHost(&bias, sizeof(DATATYPE) * oc);
  init(input, input_size);
  init(weight, weight_size);
  init(bias, oc);

  int out_size = batch * oc * oh * ow;

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight, *dev_residual, *dev_bias;
  C_DATATYPE *dev_out;


  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight, weight_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, out_size * sizeof(C_DATATYPE));
  cudaMalloc((void **)&dev_residual, output_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_bias, oc * sizeof(DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight, weight, weight_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bias, bias, oc * sizeof(DATATYPE), cudaMemcpyHostToDevice);


  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic}, {oc, kh, kw, ic}, {pad_h0, 0, pad_h0, 0},
      {stride_h, stride_w}, {1, 1}, {batch, oh, ow, oc}, mode, 1);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)dev_input, {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)dev_weight, {ic, ic * kw, ic * kw * kh}},
      {(cutlass::half_t *)dev_bias, {0, 0, 0}},
      {(cutlass::half_t *)dev_out, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f}};

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
  void *workspace = nullptr;
  assert(bytes == 0);

  cutlass::Status cutlass_status = implicit_gemm_op.can_implement(arguments);
  cutlass_check(cutlass_status);
  cutlass_status = implicit_gemm_op.initialize(arguments, workspace);
  cutlass_check(cutlass_status);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int i = 0; i < WARMUP; i++) {
    cutlass_status = implicit_gemm_op(stream);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    cutlass_status = implicit_gemm_op();
  }

  cutlass_check(cutlass_status);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("cutlass conv compute time: %f\n", elapsed_time);

  cudaStreamDestroy(stream);	
  cudaDeviceReset();

  cudaFreeHost(input);
  cudaFreeHost(weight);
  cudaFreeHost(bias);
  
  return 0;
}
