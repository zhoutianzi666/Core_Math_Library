
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 5
#define REPEATE 10

using DATATYPE = half;
using C_DATATYPE = half;

int main(void) {
  int batch = 1;
  int ic = 8;
  int ih = 224;
  int iw = 224;
  int pad_h = 1;
  int pad_w = 1;
  int oc = 32;
  int kh = 3;
  int kw = 3;
  int stride_h = 1;
  int stride_w = 1;

  // Here is consistent with
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

  // Note input and weight is in CPU place
  DATATYPE *input, *weight;
  int input_size = batch * ic * ih * iw;
  int weight_size = oc * ic * kh * kw;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  status = cudaMallocHost(&weight, sizeof(DATATYPE) * weight_size);
  init(input, input_size);
  init(weight, weight_size);

  // out is used to store the result form dev_out
  C_DATATYPE *out_from_dev;
  int out_size = batch * oc * oh * ow;
  cudaMallocHost(&out_from_dev, sizeof(C_DATATYPE) * out_size);
  memset(out_from_dev, 0, sizeof(C_DATATYPE) * out_size);

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight;
  C_DATATYPE *dev_out;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight, weight_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, out_size * sizeof(C_DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight, weight, weight_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < WARMUP; i++) {
    cutlass_nhwc_conv(dev_input, dev_weight, dev_out, batch, ic, ih, iw, kh, kw,
                      oc, pad_h, pad_w, stride_h, stride_w, oh, ow);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    cutlass_nhwc_conv(dev_input, dev_weight, dev_out, batch, ic, ih, iw, kh, kw,
                      oc, pad_h, pad_w, stride_h, stride_w, oh, ow);
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

  naive_conv_cpu(input, weight, out_cpu_fp32, batch, ic, ih, iw, kh, kw, oc,
                 pad_h, pad_w, stride_h, stride_w);

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
