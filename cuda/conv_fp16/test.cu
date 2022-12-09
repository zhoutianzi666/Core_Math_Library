#pragma once
#include <cudnn_v8.h>
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
using C_DATATYPE = half;

void CUDNN_CHECK(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("不能实施\n");
  }
}

int main(void) {
  int batch = 1;
  int ic = 3;
  int ih = 600;
  int iw = 1008;
  int pad_h = 1;
  int pad_w = 1;
  int oc = 16;
  int kh = 3;
  int kw = 3;
  int stride_h = 1;
  int stride_w = 1;

  // Here is consistent with
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

  // Note input and weight is in CPU place
  DATATYPE *input, *weight, *residual, *bias;
  int input_size = batch * ic * ih * iw;
  int weight_size = oc * ic * kh * kw;
  int output_size = batch * oc * oh * ow;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  status = cudaMallocHost(&weight, sizeof(DATATYPE) * weight_size);
  status = cudaMallocHost(&bias, sizeof(DATATYPE) * oc);
  status = cudaMallocHost(&residual, sizeof(DATATYPE) * output_size);
  init(input, input_size);
  init(weight, weight_size);
  init(residual, output_size);
  init(bias, oc);
  //  memset(bias, 0, sizeof(DATATYPE) * oc);

  // out is used to store the result form dev_out
  C_DATATYPE *out_from_dev;
  int out_size = batch * oc * oh * ow;
  cudaMallocHost(&out_from_dev, sizeof(C_DATATYPE) * out_size);
  memset(out_from_dev, 0, sizeof(C_DATATYPE) * out_size);

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight, *dev_residual, *dev_bias;
  C_DATATYPE *dev_out;

  // ---------------------------this is cuDNN-------------------------------------------------------
  
  cudnnHandle_t handle_cudnn;
  cudnnCreate(&handle_cudnn);
  cudnnTensorDescriptor_t input_descriptor;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF, batch, ic, ih, iw));

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_HALF, batch, oc, oh, ow));
  cudnnFilterDescriptor_t kernel_descriptor;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF,
                                         CUDNN_TENSOR_NCHW, oc, ic, kh, kw));

  cudnnConvolutionDescriptor_t conv_descriptor;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_descriptor));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_descriptor, pad_h, pad_w,  // zero-padding
      stride_h, stride_w,             // stride
      1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults[100];
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
      handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
      output_descriptor, 100, &returnedAlgoCount, perfResults));

  size_t workspace_size = 10000000;
  printf("%d\n", returnedAlgoCount);
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_cudnn,
                                          input_descriptor,
                                          kernel_descriptor,
                                          conv_descriptor,
                                          output_descriptor,
                                          perfResults[0].algo,
                                          &workspace_size));
  printf("%d\n", workspace_size);
  std::cout << "perfResults[0]:" << perfResults[0].algo << perfResults[0].status << std::endl;
  std::cout << "perfResults[1]:" << perfResults[1].algo << perfResults[1].status << std::endl;
  std::cout << "perfResults[2]:" << perfResults[2].algo << perfResults[2].status << std::endl;
  std::cout << "perfResults[3]:" << perfResults[3].algo << perfResults[3].status << std::endl;
  std::cout << "perfResults[4]:" << perfResults[4].algo << perfResults[4].status << std::endl;
  cudnnSetConvolutionMathType(conv_descriptor, CUDNN_TENSOR_OP_MATH);

  void *workspace = nullptr;
  cudaMalloc(&workspace, workspace_size);
  std::cout << workspace << std::endl;

  //----------------------------cuDNN ends-------------------------------------

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight, weight_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, out_size * sizeof(C_DATATYPE));
  cudaMalloc((void **)&dev_residual, output_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_bias, oc * sizeof(DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight, weight, weight_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_residual, residual, output_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bias, bias, oc * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  // ---------------------------this is also cuDNN-----------------------------------------------
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
      handle_cudnn, input_descriptor, dev_input, kernel_descriptor, dev_weight,
      conv_descriptor, output_descriptor, dev_out, 100, &returnedAlgoCount,
      perfResults, workspace, workspace_size));
  printf("返回的算法个数是：%d\n", returnedAlgoCount);

  // ------------------------cuDNN ends-----------------------------------------------------

  for (int i = 0; i < WARMUP; i++) {
    //   const float alpha = 1.0f;
    //   const float beta = 0.0f;
    //   CUDNN_CHECK(cudnnConvolutionForward(
    // handle_cudnn,
    // &alpha,
    // input_descriptor,
    // dev_input,
    // kernel_descriptor,
    // dev_weight,
    // conv_descriptor,
    // perfResults[0].algo,
    // workspace,
    // workspace_size,
    // &beta,
    // output_descriptor,
    // dev_out));

    // cutlass_nhwc_conv(dev_input, dev_weight, dev_bias, dev_out, batch, ic,
    // ih,
    //                   iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w, oh,
    //                   ow);

    cutlass_nhwc_conv_bias_swish(dev_input, dev_weight, dev_bias, dev_out,
                                 batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
                                 stride_h, stride_w, oh, ow);

    //  cutlass_nhwc_conv_bias_leaky_relu(dev_input, dev_weight, dev_bias, dev_out,
    //   batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    //   stride_h, stride_w, oh, ow);

    // my_implicit_gemm_gpu(dev_input, dev_weight, dev_bias, dev_out,
    // batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    // stride_h, stride_w, oh, ow);

    // cutlass_nhwc_conv_bias_swish_simt(dev_input, dev_weight, dev_bias, dev_out,
    //   batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    //   stride_h, stride_w, oh, ow);

    
    // my_naive_conv_gpu(dev_input, dev_weight, dev_bias, dev_out, batch, ic,
    // ih, iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w, oh, ow);

    // cutlass_nhwc_conv_residual(dev_input, dev_weight, dev_bias, dev_out,
    // batch, ic, ih, iw, kh, kw,
    //   oc, pad_h, pad_w, stride_h, stride_w, oh, ow, dev_residual);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    // const float alpha = 1.0f;
    // const float beta = 0.0f;
    // cudnnConvolutionForward(
    // handle_cudnn,
    // &alpha,
    // input_descriptor,
    // dev_input,
    // kernel_descriptor,
    // dev_weight,
    // conv_descriptor,
    // perfResults[0].algo,
    // workspace,
    // workspace_size,
    // &beta,
    // output_descriptor,
    // dev_out);

    // cutlass_nhwc_conv(dev_input, dev_weight, dev_bias, dev_out, batch, ic,
    // ih,
    //                   iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w, oh,
    //                   ow);

    cutlass_nhwc_conv_bias_swish(dev_input, dev_weight, dev_bias, dev_out,
                                 batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
                                 stride_h, stride_w, oh, ow);

    // cutlass_nhwc_conv_bias_leaky_relu(dev_input, dev_weight, dev_bias, dev_out,
    //   batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    //   stride_h, stride_w, oh, ow);

    // my_implicit_gemm_gpu(dev_input, dev_weight, dev_bias, dev_out,
    // batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    // stride_h, stride_w, oh, ow);

    // cutlass_nhwc_conv_bias_swish_simt(dev_input, dev_weight, dev_bias, dev_out,
    //   batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    //   stride_h, stride_w, oh, ow);
    
    // my_naive_conv_gpu(dev_input, dev_weight, dev_bias, dev_out, batch, ic,
    // ih,
    //     iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w, oh, ow);

    // cutlass_nhwc_conv_residual(dev_input, dev_weight, dev_bias, dev_out,
    // batch, ic, ih, iw, kh, kw,
    //   oc, pad_h, pad_w, stride_h, stride_w, oh, ow, dev_residual);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu conv compute time: %f\n", elapsed_time);
  double Gflops =
      REPEATE * ((float)out_size * ic * kh * kw * 2 / 1000000) / elapsed_time;
  printf("Gflops: %5.2f \n", Gflops);

  cudaMemcpy(out_from_dev, dev_out, out_size * sizeof(C_DATATYPE),
             cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();

  // results calculated in cpu is always fp32
  float *out_cpu_fp32 = (float *)malloc(sizeof(float) * out_size);
  memset(out_cpu_fp32, 0, sizeof(float) * out_size);

  naive_conv_cpu(input, weight, bias, out_cpu_fp32, batch, ic, ih, iw, kh, kw,
                 oc, pad_h, pad_w, stride_h, stride_w, nullptr);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(out_from_dev, out_cpu_fp32, out_size));

  cudaDeviceReset();
  cudaFreeHost(input);
  cudaFreeHost(weight);
  cudaFreeHost(out_from_dev);
  free(out_cpu_fp32);
  return 0;
}
