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
# define M 8
__global__ void kernel1(DATATYPE *a, int n, DATATYPE *c) {
  const int tidx = threadIdx.x;
  int group_id = threadIdx.x / M;
  
  DATATYPE qk = a[tidx];
  
  if (tidx <= 32) {

    for (int mask = M / 2; mask >= 1; mask /= 2) {
        qk += __shfl_xor_sync(FULL_MASK, qk, mask);
    }
    if (tidx % M == 0)
    c[group_id] = qk;

  }
}

int main(void) {
  // 32个数字，每4个数字求和，放到每组的第0个数字！
  int n = 32;
  DATATYPE *a;
  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * n);
  if (status != cudaSuccess) {
    printf("分配paged内存失败");
  }
  init(a, n);
  
  int group_num = n / M;
  DATATYPE c_from_gpu[group_num]={0};
  DATATYPE c_result_in_cpu[group_num]={0};

  // allocate the memory on the GPU
  DATATYPE *dev_a, *dev_c;
  cudaMalloc((void **)&dev_a, n * sizeof(DATATYPE));
  cudaMemcpy(dev_a, a, n * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&dev_c, group_num * sizeof(DATATYPE));
  cudaMemset(dev_c, 0, group_num * sizeof(DATATYPE));

  kernel1<<<1, 32>>>(dev_a, n, dev_c);


  cudaMemcpy(&c_from_gpu, dev_c, group_num * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

  for (int i = 0; i < n; i+= M) {
    
    for (int j = 0; j < M; j++) {
        c_result_in_cpu[i/M] += a[i + j];
    }
  }
  for (int j = 0; j < group_num; j++) {
    printf("%f\n", c_result_in_cpu[j]);
    printf("%f\n", c_from_gpu[j]);
  }

  cudaDeviceReset();
  cudaFreeHost(a);
  return 0;
}
