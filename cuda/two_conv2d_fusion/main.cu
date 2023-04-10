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
#define REPEATE 100

using DATATYPE = half;
using C_DATATYPE = half;

void CUDNN_CHECK(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("CUDNN 不能实施\n");
  }
}

int main(void) {
  int batch0 = 1;
  int ic0 = 256;
  int ih0 = 80;
  int iw0 = 80;
  int pad0_h0 = 0;
  int pad0_h1 = 0;
  int pad0_w0 = 0;
  int pad0_w1 = 0;
  int oc0 = 64;
  int kh0 = 1;
  int kw0 = 1;
  int stride0_h = 1;
  int stride0_w = 1;
  int dilation0_h = 1;
  int dilation0_w = 1;

  // Here is consistent with
  int oh0 = (ih0 + pad0_h0 + pad0_h1 - dilation0_h * (kh0 - 1) - 1) / stride0_h + 1;
  int ow0 = (iw0 + pad0_w0 + pad0_w1 - dilation0_w * (kw0 - 1) - 1) / stride0_w + 1;

  int batch1 = batch0;
  int ic1 = oc0;
  int ih1 = oh0;
  int iw1 = oh0;
  int pad1_h0 = 0;
  int pad1_h1 = 0;
  int pad1_w0 = 0;
  int pad1_w1 = 0;
  int oc1 = 64;
  int kh1 = 1;
  int kw1 = 1;
  int stride1_h = 1;
  int stride1_w = 1;
  int dilation1_h = 1;
  int dilation1_w = 1;

  int oh1 = (ih1 + pad1_h0 + pad1_h1 - dilation1_h * (kh1 - 1) - 1) / stride1_h + 1;
  int ow1 = (iw1 + pad1_w0 + pad1_w1 - dilation1_w * (kw1 - 1) - 1) / stride1_w + 1;


  // Note input and weight is in CPU place
  DATATYPE *input, *weight0, *bias0, *weight1, *bias1;
  int input_size = batch0 * ic0 * ih0 * iw0;
  int weight0_size = oc0 * ic0 * kh0 * kw0;
  int bias0_size = oc0;
  int weight1_size = oc1 * ic1 * kh1 * kw1;
  int bias1_size = oc1;
  DATATYPE* b2b_scale, *b2b_bias;

  input = (DATATYPE *)malloc(sizeof(DATATYPE) * input_size);
  weight0 = (DATATYPE *)malloc(sizeof(DATATYPE) * weight0_size);
  weight1 = (DATATYPE *)malloc(sizeof(DATATYPE) * weight1_size);
  bias0 = (DATATYPE *)malloc(sizeof(DATATYPE) * bias0_size);
  bias1 = (DATATYPE *)malloc(sizeof(DATATYPE) * bias1_size);
  
  b2b_scale = (DATATYPE *)malloc(sizeof(DATATYPE) * bias0_size);
  b2b_bias = (DATATYPE *)malloc(sizeof(DATATYPE) * bias0_size);


  init(input, input_size);
  init(weight0, weight0_size);
  init(bias0, bias0_size);
  init(weight1, weight1_size);
  init(bias1, bias1_size);
  init(b2b_scale, bias0_size);
  init(b2b_bias,  bias0_size);

  for (int i = 0; i < bias0_size; i++)
  {
    b2b_bias[i] = 0.f;
    b2b_scale[i] = 1.f;
  }


  // out_from_dev is used to store the result form dev_out
  C_DATATYPE *out_from_dev;

  int out0_size = batch0 * oc0 * oh0 * ow0;
  int out1_size = batch1 * oc1 * oh1 * ow1;

  out_from_dev = (C_DATATYPE *)malloc(sizeof(C_DATATYPE) * out1_size);
  memset(out_from_dev, 0, sizeof(C_DATATYPE) * out1_size);

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight0, *dev_bias0, *dev_weight1, *dev_bias1;
  DATATYPE *dev_b2b_scale, *dev_b2b_bias;
  C_DATATYPE *dev_out;

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight0, weight0_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight1, weight1_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_bias0, bias0_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_bias1, bias1_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b2b_scale, bias0_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b2b_bias, bias0_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, out1_size * sizeof(C_DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight0, weight0, weight0_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight1, weight1, weight1_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bias0, bias0, bias0_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bias1, bias1, bias1_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b2b_scale, b2b_scale, bias0_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b2b_bias, b2b_bias,   bias0_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);


  ConvAllParams p0;
  p0.batch = batch0;
  p0.ic = ic0;
  p0.ih = ih0;
  p0.iw = iw0;
  p0.pad_h0 = pad0_h0;
  p0.pad_h1 = pad0_h1;
  p0.pad_w0 = pad0_w0;
  p0.pad_w1 = pad0_w1;
  p0.oc = oc0;
  p0.kh = kh0;
  p0.kw = kw0;
  p0.stride_h = stride0_h;
  p0.stride_w = stride0_w;
  p0.dilation_h = dilation0_h;
  p0.dilation_w = dilation0_w;
  p0.oh = oh0;
  p0.ow = ow0;
  p0.input = dev_input;
  p0.weight = dev_weight0;
  p0.bias = dev_bias0;
  p0.residual = nullptr;
  p0.output = nullptr;
  p0.b2b_scale = dev_b2b_scale;
  p0.b2b_bias = dev_b2b_bias;
  p0.act_type = RELU;


  ConvAllParams p1;
  p1.batch = batch1;
  p1.ic = ic1;
  p1.ih = ih1;
  p1.iw = iw1;
  p1.pad_h0 = pad1_h0;
  p1.pad_h1 = pad1_h1;
  p1.pad_w0 = pad1_w0;
  p1.pad_w1 = pad1_w1;
  p1.oc = oc1;
  p1.kh = kh1;
  p1.kw = kw1;
  p1.stride_h = stride1_h;
  p1.stride_w = stride1_w;
  p1.dilation_h = dilation1_h;
  p1.dilation_w = dilation1_w;
  p1.oh = oh1;
  p1.ow = ow1;
  p1.input = nullptr;
  p1.weight = dev_weight1;
  p1.bias = dev_bias1;
  p1.residual = nullptr;
  p1.output = dev_out;
  p1.act_type = IDENTITY;

  for (int i = 0; i < WARMUP; i++) {
    cutlass_nhwc_conv_bias(p0, p1);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    cutlass_nhwc_conv_bias(p0, p1);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu conv compute time: %f\n", elapsed_time);
  double Gflops = REPEATE * ((float) 0  * 2 / 1000000) / elapsed_time;
  printf("Gflops: %5.2f \n", Gflops);

  cudaMemcpy(out_from_dev, dev_out, out1_size * sizeof(C_DATATYPE),
             cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();

  // results calculated in cpu is always fp32
  float *out0_cpu_fp32 = (float *)malloc(sizeof(float) * out0_size);
  float *out1_cpu_fp32 = (float *)malloc(sizeof(float) * out1_size);
  memset(out0_cpu_fp32, 0, sizeof(float) * out0_size);
  memset(out1_cpu_fp32, 0, sizeof(float) * out1_size);

  p0.input = input;
  p0.weight = weight0;
  p0.bias = bias0;
  p0.output_cpu_fp32 = out0_cpu_fp32;
  p0.residual = nullptr;
  naive_conv_cpu(p0, half(0.f));
  p1.input = (half*)out0_cpu_fp32;
  p1.weight = weight1;
  p1.bias = bias1;
  p1.output_cpu_fp32 = out1_cpu_fp32;
  p1.residual = nullptr;
  naive_conv_cpu(p1, float(0.f));


  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(out_from_dev, out1_cpu_fp32, out1_size));

  cudaDeviceReset();
  free(input);
  free(weight0);
  free(weight1);
  free(bias0);
  free(bias1);
  free(out_from_dev);
  free(out0_cpu_fp32);
  free(out1_cpu_fp32);
  return 0;
}
