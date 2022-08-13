#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#define FULL_MASK 0xffffffff
using DATATYPE = float;

void init(DATATYPE *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (rand() % 9999) / 10000.0;
  }
}

__global__ void reduce1(DATATYPE *a, int n, DATATYPE *c) {
  const int tidx = threadIdx.x;
  DATATYPE val = a[tidx];
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_MASK, val, offset);
  }
  if (tidx == 0) {
    *c = val;
  }
}

int main(void) {
  int n = 32;
  DATATYPE *a;
  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * n);
  if (status != cudaSuccess) {
    printf("分配paged内存失败");
  }
  init(a, n);

  DATATYPE c_from_gpu;

  // allocate the memory on the GPU
  DATATYPE *dev_a, *dev_c;
  cudaMalloc((void **)&dev_a, n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_c, sizeof(DATATYPE));
  cudaMemcpy(dev_a, a, n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  uint3 grid = {1, 1, 1};
  uint3 block = {32, 1, 1};

  reduce1<<<grid, block>>>(dev_a, n, dev_c);

  cudaMemcpy(&c_from_gpu, dev_c, sizeof(DATATYPE), cudaMemcpyDeviceToHost);

  float c_result_in_cpu = 0;
  for (int i = 0; i < n; i++) {
    c_result_in_cpu += a[i];
  }
  printf("%f\n", c_result_in_cpu);
  printf("%f\n", c_from_gpu);

  cudaDeviceReset();
  cudaFreeHost(a);
  return 0;
}
