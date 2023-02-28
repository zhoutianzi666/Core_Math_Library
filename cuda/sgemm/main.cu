
#include <stdio.h>
#include <assert.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 0
#define REPEATE 1

using DATATYPE = float;
using C_DATATYPE = float;

int main(void) {
  int m = 5;
  int n = 5;
  int k = 9; 
  cudaSetDevice(3);

  DATATYPE *a, *b;
  a = (DATATYPE *)malloc(sizeof(DATATYPE) * m * k);
  b = (DATATYPE *)malloc(sizeof(DATATYPE) * k * n);
  assert(a);
  assert(b);
  init(a, m * k);
  init(b, k * n);

for (int i = 0; i < m * k ;i++)
{
  a[i] = i;
}

for (int i = 0; i < k * n;i++)
{
  b[i] = i;
}

  C_DATATYPE *c;
  c = (C_DATATYPE *)malloc(sizeof(C_DATATYPE) * m * n);
  assert(c);
  memset(c, 0, sizeof(C_DATATYPE) * m * n);

  DATATYPE *dev_a, *dev_b;
  C_DATATYPE *dev_c;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(C_DATATYPE));

  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  for (int i = 0; i < WARMUP; i++) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CutlassSgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
    // cublas_matmul(handle, dev_a, dev_b, dev_c, m, n , k);
    // matmul_gpu(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_megengine(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_naive_block(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_naive_block_combine_access(dev_a, dev_b, dev_c, m, n, k);
    //matmul_gpu_naive(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CutlassSgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
    // cublas_matmul(handle, dev_a, dev_b, dev_c, m, n , k);
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n,
    //             dev_a, k, &beta, dev_c, n);
    // matmul_gpu(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_megengine(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_naive_block(dev_a, dev_b, dev_c, m, n, k);
    // matmul_gpu_naive_block_combine_access(dev_a, dev_b, dev_c, m, n, k);
    //matmul_gpu_naive(dev_a, dev_b, dev_c, m, n, k);
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

  float *c_cpu_fp32 = (float *)malloc(sizeof(float) * m * n);
  memset(c_cpu_fp32, 0, sizeof(float) * m * n);
  naive_gemm_cpu(a, b, c_cpu_fp32, m, n, k);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(c, c_cpu_fp32, m * n));

  cudaDeviceReset();
  free(a);
  free(b);
  free(c);
  free(c_cpu_fp32);
  return 0;
}
