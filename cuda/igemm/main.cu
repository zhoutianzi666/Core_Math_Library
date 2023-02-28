
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

void CUDA_CHECK(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("分配paged内存失败\n");
  }
}

int main(void) {
  int m = 512;
  int n = 512;
  int k = 512;

  DATATYPE *a, *b;
  BIAS_DATATYPE *bias;

  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  CUDA_CHECK(status);
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);
  CUDA_CHECK(status);
  status = cudaMallocHost(&bias, sizeof(BIAS_DATATYPE) * n);
  CUDA_CHECK(status);

  init(a, m * k);
  init(b, k * n);
  init(bias, k * n);

  C_DATATYPE *c;
  cudaMallocHost(&c, sizeof(C_DATATYPE) * m * n);
  memset(c, 0, sizeof(C_DATATYPE) * m * n);

  DATATYPE *dev_a, *dev_b;
  BIAS_DATATYPE *dev_bias;
  C_DATATYPE *dev_c;
  
  cublasHandle_t handle;
  cublasCreate(&handle);

  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();
  
 // allocate the memory on the GPU and copy a and b to GPU
  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_bias, n * sizeof(BIAS_DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(C_DATATYPE));

  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bias, bias, n * sizeof(BIAS_DATATYPE), cudaMemcpyHostToDevice);

  for (int i = 0; i < WARMUP; i++) {
    CutlassIgemmNN(n, m, k, dev_a, k, dev_b, k, dev_bias, dev_c, n);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    CutlassIgemmNN(n, m, k, dev_a, k, dev_b, k, dev_bias, dev_c, n);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu gemm compute time: %f\n", elapsed_time);
  double Gflops = REPEATE * ((float)m * n * k * 2 / 1000000) / elapsed_time;
  printf("Gflops: %5.2f \n", Gflops);

  cudaMemcpy(c, dev_c, m * n * sizeof(C_DATATYPE), cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();

  int32_t *c_cpu_int32 = (int32_t *)malloc(sizeof(float) * m * n);
  memset(c_cpu_int32, 0, sizeof(int32_t) * m * n);
  naive_gemm_cpu(a, b, c_cpu_int32, m, n, k);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %d\n", diff(c, c_cpu_int32, m * n));

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu_int32);
  return 0;
}
