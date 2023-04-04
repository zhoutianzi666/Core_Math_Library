#pragma once
#include <cudnn_v8.h>
#include <stdio.h>
#include <assert.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 100

using DATATYPE = half;
using C_DATATYPE = half;

int main(void) {
  //cudaSetDevice(3);
  int batch = 1;
  int ic = 32;
  int ih = 112;
  int iw = 112;
  int pad_h0 = 1;
  int pad_h1 = 1;
  int pad_w0 = 1;
  int pad_w1 = 1;
  int groups = 32;
  int kc = ic / groups;
  int oc = 32;
  int kh = 3;
  int kw = 3;
  int stride_h = 1;
  int stride_w = 1;
  int dilation_h = 1;
  int dilation_w = 1;

  // Here is consistent with
  int oh = (ih + pad_h0 + pad_h1 - dilation_h * (kh - 1) - 1) / stride_h + 1;
  int ow = (iw + pad_w0 + pad_w1 - dilation_w * (kw - 1) - 1) / stride_w + 1;


  ConvAllParams params;

  // Note input and weight is in CPU place
  DATATYPE *input, *weight, *residual, *bias;
  int input_size = batch * ic * ih * iw;
  int weight_size = oc * kc * kh * kw;
  int output_size = batch * oc * oh * ow;

  cudaError_t status ;
  input = (DATATYPE *)malloc(sizeof(DATATYPE) * input_size);
  weight = (DATATYPE *)malloc(sizeof(DATATYPE) * weight_size);
  bias = (DATATYPE *)malloc(sizeof(DATATYPE) * oc);
  residual = (DATATYPE *)malloc(sizeof(DATATYPE) * output_size);
  assert(input);
  assert(weight);
  assert(bias);
  assert(residual);

  init(input, input_size);
  init(weight, weight_size);
  init(residual, output_size);
  init(bias, oc);
  //memset(bias, 0, sizeof(C_DATATYPE) * oc);

  // out is used to store the result form dev_out
  C_DATATYPE *out_from_dev;
  int out_size = batch * oc * oh * ow;
  out_from_dev = (C_DATATYPE*)malloc(sizeof(C_DATATYPE) * out_size);
  memset(out_from_dev, 0, sizeof(C_DATATYPE) * out_size);

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight, *dev_residual, *dev_bias;
  C_DATATYPE *dev_out;

  // ---------------------------this is cuDNN-------------------------------------------------------

  cudnnHandle_t handle_cudnn;
  cudnnCreate(&handle_cudnn);
  params.handle_cudnn = handle_cudnn;
  
  cudnnTensorDescriptor_t input_descriptor;
  auto cudnn_layout = CUDNN_TENSOR_NHWC;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, cudnn_layout,
                                         CUDNN_DATA_HALF, batch, ic, ih, iw));

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, cudnn_layout,
                                         CUDNN_DATA_HALF, batch, oc, oh, ow));
  cudnnFilterDescriptor_t kernel_descriptor;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF,
                                         cudnn_layout, oc, kc, kh, kw));

  cudnnTensorDescriptor_t bias_descriptor;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_descriptor, cudnn_layout,
                                         CUDNN_DATA_HALF,
                                         1, oc, 1, 1));

  cudnnConvolutionDescriptor_t conv_descriptor;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_descriptor));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_descriptor, pad_h0, pad_w0,  // 右边不需要告诉他,还是默认是对称的呢？
      stride_h, stride_w,               // stride
      dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_descriptor, groups));

  cudnnActivationDescriptor_t act_desc;
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
  cudnnSetActivationDescriptor(
    act_desc,
    CUDNN_ACTIVATION_IDENTITY,
    CUDNN_PROPAGATE_NAN,
    0.f
  );


  std::cout << "这个我猜可能是逻辑判断来确定的最优算法！" << std::endl;
  int returnedAlgoCount;
  cudnnConvolutionFwdAlgoPerf_t perfResults[100];
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
      handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
      output_descriptor, 100, &returnedAlgoCount, perfResults));

  size_t cudnn_workspace_size = 0;
  printf("\t返回的算法个数是：%d\n", returnedAlgoCount);
  for (int i = 0; i < returnedAlgoCount; i++) {
  std::cout << "\t perfResults[" << i<< "]:" << cudnnAlgoName(perfResults[i].algo) 
            <<" " << perfResults[i].status
            << std::endl;
  }
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
      output_descriptor, perfResults[0].algo, &cudnn_workspace_size));
  printf("\t cudnn_workspace_size: %d\n", cudnn_workspace_size);
  std::cout << "\t启发式搜索想要的空间大小为: " << cudnn_workspace_size<< std::endl;
  cudnn_workspace_size = 150 * 1000 * 1000;
  printf("\t我将其设置为 cudnn_workspace_size: %d\n", cudnn_workspace_size); 

  cudnnSetConvolutionMathType(conv_descriptor, CUDNN_TENSOR_OP_MATH);

  void *cudnn_workspace = nullptr;
  cudaMalloc(&cudnn_workspace, cudnn_workspace_size);
  std::cout << "\tcudnn_workspace pointer: " << cudnn_workspace << std::endl;

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
  std::cout << "这个我猜可能是真实的计算来确定的最优算法！" << std::endl;
  CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
      handle_cudnn, input_descriptor, dev_input, kernel_descriptor, dev_weight,
      conv_descriptor, output_descriptor, dev_out, 100, &returnedAlgoCount,
      perfResults, cudnn_workspace, cudnn_workspace_size));
  for (int i = 0; i < returnedAlgoCount; i++) {
  std::cout << "\tperfResults[" << i<< "]:" << cudnnAlgoName(perfResults[i].algo) 
            <<" " << perfResults[i].status
            << std::endl;
  }
  printf("\t返回的算法个数是：%d\n", returnedAlgoCount);
  cudaMemset(dev_out, 0, sizeof(C_DATATYPE) * out_size);

  // 上面必须要清零！因为cuDNN要计算这个结果！

  // ------------------------cuDNN ends-----------------------------------------------------

  params.batch = batch;
  params.ic = ic;
  params.ih = ih;
  params.iw = iw;
  params.pad_h0 = pad_h0;
  params.pad_h1 = pad_h1;
  params.pad_w0 = pad_w0;
  params.pad_w1 = pad_w1;
  params.oc = oc;
  params.kh = kh;
  params.kw = kw;
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.dilation_h = dilation_h;
  params.dilation_w = dilation_w;
  params.groups = groups;

  // Here is consistent with
  params.oh = oh;
  params.ow = ow;

  params.input = dev_input;
  params.weight = dev_weight;
  params.residual = dev_residual;
  params.bias = dev_bias;
  params.output = dev_out;

  cudaEvent_t beg, end;


  params.cudnn_workspace_size = 150 * 1000 * 1000;
  cudaMalloc(&params.cudnn_workspace, params.cudnn_workspace_size);

  for (int i = 0; i < WARMUP + REPEATE; i++) {

    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }

    params.act_type = IDENTITY;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 下面是conv+add
    // CUDNN_CHECK(cudnnConvolutionForward(
    // handle_cudnn,
    // &alpha, 
    // input_descriptor, dev_input,
    // kernel_descriptor, dev_weight,
    // conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, cudnn_workspace, cudnn_workspace_size, &beta, 
    // output_descriptor, dev_out));
    // CUDNN_CHECK(cudnnAddTensor(handle_cudnn, &alpha, 
    // bias_descriptor, dev_bias, &alpha, output_descriptor, dev_out));
    // 这个是激活
    // CUDNN_CHECK(cudnnActivationForward(
    // handle_cudnn,
    // act_desc,
    // &alpha,
    // output_descriptor,
    // dev_out,
    // &beta,
    // output_descriptor,
    // dev_out));
  
  // 这个是cba
  //  CUDNN_CHECK(cudnnConvolutionBiasActivationForward(handle_cudnn,
  //   &alpha,
  //   input_descriptor,
  //   dev_input,
  //   kernel_descriptor,
  //   dev_weight,
  //   conv_descriptor,
  //   CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
  //   cudnn_workspace,
  //   cudnn_workspace_size,
  //   &beta,
  //   output_descriptor,
  //   dev_out,
  //   bias_descriptor,
  //   dev_bias,
  //   act_desc,
  //   output_descriptor,
  //   dev_out));

    //这个函数里面包含着各种创建descriptor的cudnn kernel。
    // params.act_type = IDENTITY;
    // cudnn_nhwc_conv(params);

    // params.act_type = IDENTITY;
    // cutlass_nhwc_conv_relu(params);
    
    // params.act_type = SILU;
    // cutlass_nhwc_conv_bias_swish(params);
    // cutlass_nhwc_conv_bias_swish_simt(params);

    // params.act_type = LEAKY_RELU;
    // cutlass_nhwc_conv_bias_leaky_relu(params);

    // my_implicit_gemm_gpu(dev_input, dev_weight, dev_bias, dev_out,
    // batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w,
    // stride_h, stride_w, oh, ow);

    // my_naive_conv_gpu(dev_input, dev_weight, dev_bias, dev_out, batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w, oh, ow);

    // params.act_type = CONV2D_BIAS_ADD_RELU;
    // cutlass_nhwc_conv_residual(params);

    params.act_type = IDENTITY;
    cutlass_nhwc_conv_depthwise(params);
    
    // params.act_type = SILU;
    // cutlass_group_conv_bias_swish(params);
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

  params.input = input;
  params.weight = weight;
  params.bias = bias;
  params.output_cpu_fp32 = out_cpu_fp32;
  params.residual = residual;

  naive_conv_cpu(params);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(out_from_dev, out_cpu_fp32, out_size));

  cudaDeviceReset();
  free(input);
  free(weight);
  free(out_from_dev);
  free(out_cpu_fp32);
  
  cudaFree(params.cudnn_workspace);
  cudaFree(cudnn_workspace);
  return 0;
}
