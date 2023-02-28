#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;

int main(void) {
  int batch = 1;
  int ic = 32;
  int ih = 1;
  int iw = 32;
  // Note input is in CPU place
  DATATYPE *input;
  int input_size = batch * ic * ih * iw;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  init(input, input_size);

  // out is used to store the result form dev_out
  DATATYPE *out_from_dev;
  int out_size = input_size;
  cudaMallocHost(&out_from_dev, sizeof(DATATYPE) * out_size);
  memset(out_from_dev, 0, sizeof(DATATYPE) * out_size);

  DATATYPE *dev_input, *dev_out;

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, input_size * sizeof(DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);

  for (int i = 0; i < WARMUP; i++) {
    //my_naive_nchw_nhwc(dev_input, dev_out, batch, ic, ih * iw);
    //cutlass_nchw_nhwc(dev_input, dev_out, batch, ic, ih, iw);
    //cutlass_nhwc_nchw(dev_input, dev_out, batch, ic, ih, iw);
    //cudaMemcpy(dev_out, dev_input, sizeof(half) * out_size, cudaMemcpyDeviceToDevice);
    //my_row_col0(dev_out, dev_input, batch, ic, ih * iw);
    //my_row_col1(dev_out, dev_input, batch, ic, ih * iw);
    cutlass_iter(dev_out, dev_input, batch, ic, ih * iw);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    //my_naive_nchw_nhwc(dev_input, dev_out, batch, ic, ih * iw);
    //cutlass_nchw_nhwc(dev_input, dev_out, batch, ic, ih, iw);
    //cutlass_nhwc_nchw(dev_input, dev_out, batch, ic, ih, iw);
    // cudaMemcpy(dev_out, dev_input, sizeof(half) * out_size, cudaMemcpyDeviceToDevice);
    //my_row_col0(dev_out, dev_input, batch, ic, ih * iw);
    //my_row_col1(dev_out, dev_input, batch, ic, ih * iw);
    cutlass_iter(dev_out, dev_input, batch, ic, ih * iw);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu layout time: %f\n", elapsed_time);

  cudaMemcpy(out_from_dev, dev_out, out_size * sizeof(DATATYPE),
             cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();

  // results calculated in cpu is always fp32
  half *out_cpu_fp16 = (half *)malloc(sizeof(half) * out_size);
  memset(out_cpu_fp16, 0, sizeof(half) * out_size);

  naive_nchw_nhwc_cpu(input, out_cpu_fp16, batch, ic, ih, iw);
  //naive_nhwc_nchw_cpu(input, out_cpu_fp16, batch, ic, ih, iw);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(out_from_dev, out_cpu_fp16, out_size));

  cudaDeviceReset();
  cudaFreeHost(input);
  free(out_cpu_fp16);
  return 0;
}
