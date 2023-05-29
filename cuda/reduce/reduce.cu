#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

using DATATYPE = float;

void init(DATATYPE *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (rand() % 9999) / 10000.0;
  }
}



#define FULL_MASK 0xffffffff
#define warp_size 32
#define warp_nums_in_block 32
__global__ void kernel1(DATATYPE *a, int n, DATATYPE *c) {
  const int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  float sum = 0.f;
  __shared__ float smem[warp_nums_in_block];

  for (int i = tidx; i < n; i += blockDim.x * gridDim.x) {
    sum += a[i];
  }

  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_down_sync(FULL_MASK, sum, offset);
  }
  
  if (threadIdx.x % warp_size == 0) {
    smem[threadIdx.x / warp_size] = sum;
  }
  
  sum = 0.f;
  if(threadIdx.x == 0) {
    // 这里用非常CPU的方式哈哈哈哈哈哈！
    for(int i = 0; i < warp_nums_in_block; i++)
    {
      sum += smem[i];
    }
  }
  
  if (tidx % blockDim.x == 0) {
    atomicAdd(c, sum);
  }
}

void many_block_reduce(DATATYPE *dev_a, int n, DATATYPE *dev_c) {
    uint3 grid = {32, 1, 1};
    uint3 block = {1024, 1, 1};
    cudaMemset(dev_c, 0, sizeof(DATATYPE));
    kernel1<<<grid, block>>>(dev_a, n, dev_c);
}


// __global__ void reduce1(DATATYPE *a, int n, DATATYPE *c) {
//   const int tidx = threadIdx.x;
//   DATATYPE val = a[tidx];
//   for (int offset = 16; offset > 0; offset /= 2) {
//     val += __shfl_down_sync(FULL_MASK, val, offset);
//   }
//   if (tidx == 0) {
//     *c = val;
//   }
// }



int main(void) {
  int n = 64000;
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
  cudaMemset(dev_c, 0, sizeof(DATATYPE));
  cudaMemcpy(dev_a, a, n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  constexpr int WARMUP =  10;
  constexpr int REPEATE =  100;

  cudaEvent_t beg, end;

  for (int i = 0; i < WARMUP + REPEATE; i++) {

    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }
    many_block_reduce(dev_a, n, dev_c);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu conv compute time: %f\n", elapsed_time);

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
