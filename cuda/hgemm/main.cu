
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

void CUDA_CHECK(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("分配paged内存失败\n");
  }
}

int main(void) {
  int m = 512;
  int n = 512;
  int k = 512;
  // a,b,c is in cpu place!
  DATATYPE *a, *b, *broadcast;
  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  CUDA_CHECK(status);
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);
  CUDA_CHECK(status);
  status = cudaMallocHost(&broadcast, sizeof(DATATYPE) * n);
  CUDA_CHECK(status);

  init(a, m * k);
  init(b, k * n);
  init(broadcast, n);

  C_DATATYPE *c;
  status = cudaMallocHost(&c, sizeof(C_DATATYPE) * m * n);
  CUDA_CHECK(status);
  memset(c, 0, sizeof(C_DATATYPE) * m * n);

  DATATYPE *dev_a, *dev_b, *dev_broadcast;
  C_DATATYPE *dev_c;

  // init cublas handle
  cublasHandle_t handle;
  cublasCreate(&handle);

  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  // allocate the memory on the GPU and copy a and b to GPU
  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_broadcast, n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(C_DATATYPE));
  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_broadcast, broadcast, n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  cudaEvent_t beg, end;

  for (int i = 0; i < REPEATE + REPEATE; i++) {


    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }

    const DATATYPE alpha = 1.0f;
    const DATATYPE beta = 0.0f;

    // 而且这里带有激活函数哦，这里是relu
    // GemmWithBroadcast(m, n, k, alpha, dev_a, k, dev_b, n, beta, dev_c, n, dev_broadcast);

    CutlassHgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
    // cublasHgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
    //                           n,m,k,
    //                           &alpha,
    //                           dev_b,n,
    //                           dev_a,k,
    //                           &beta,
    //                           dev_c,n);
    // matmul_gpu(dev_a, dev_b, dev_c, m, n, k);
    //   matmul_gpu_mma(dev_a, dev_b, dev_c, m, n, k);
    //    matmul_gpu_naive_block(dev_a, dev_b, dev_c, m, n, k);
    //   matmul_gpu_naive_block_combine_access(dev_a, dev_b, dev_c, m, n, k);
    //  matmul_gpu_naive(dev_a, dev_b, dev_c, m, n, k);
    // matmul_wmma(dev_a, dev_b, dev_c, m, n, k);
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
  //naive_gemm_cpu(a, b, c_cpu_fp32, m, n, k, broadcast, "relu");
  naive_gemm_cpu(a, b, c_cpu_fp32, m, n, k, nullptr, "identity");

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(c, c_cpu_fp32, m * n));

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu_fp32);
  return 0;
}
