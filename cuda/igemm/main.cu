
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
  BROADCAST_DATATYPE *broadcast;

  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  CUDA_CHECK(status);
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);
  CUDA_CHECK(status);
  status = cudaMallocHost(&broadcast, sizeof(BROADCAST_DATATYPE) * n);
  CUDA_CHECK(status);

  init(a, m * k);
  init(b, k * n);
  init(broadcast, n);

  C_DATATYPE *c;
  cudaMallocHost(&c, sizeof(C_DATATYPE) * m * n);
  memset(c, 0, sizeof(C_DATATYPE) * m * n);

  DATATYPE *dev_a, *dev_b;
  BROADCAST_DATATYPE *dev_broadcast;
  C_DATATYPE *dev_c;
  
  cublasHandle_t handle;
  cublasCreate(&handle);

  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();
  
 // allocate the memory on the GPU and copy a and b to GPU
  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_broadcast, n * sizeof(BROADCAST_DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(C_DATATYPE));

  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_broadcast, broadcast, n * sizeof(BROADCAST_DATATYPE), cudaMemcpyHostToDevice);

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < WARMUP + REPEATE; i++) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }
    CutlassIgemmNN(n, m, k, dev_a, k, dev_b, k, dev_broadcast, dev_c, n);
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
  

  // 这个是CPU上的baseline的数据类型！
  // 输出要么是fp32，要么是int32！
  using C_base_DATATYPE = float;
  C_base_DATATYPE *c_cpu_32 = (C_base_DATATYPE *)malloc(sizeof(C_base_DATATYPE) * m * n);
  memset(c_cpu_32, 0, sizeof(C_base_DATATYPE) * m * n);
  naive_gemm_cpu(a, b, c_cpu_32, m, n, k, broadcast);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);
  
  std::cout << "max_diff:" << diff(c, c_cpu_32, m * n)  << std::endl;

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  cudaFreeHost(broadcast);
  free(c_cpu_32);
  return 0;
}
