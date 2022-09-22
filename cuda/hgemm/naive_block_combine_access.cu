#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = float;
#define DATATYPE_BYTE 4

using ACCU_DATATYPE = float;

#define block_M 32
#define block_N 32
#define block_K 32
__global__ void kernel_naive_block_combine_access(DATATYPE *a, DATATYPE *b,
                                                  DATATYPE *c, int m, int n,
                                                  int k) {
  const int col = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = threadIdx.y + blockIdx.y * blockDim.y;
  __shared__ DATATYPE aTile[block_M][block_K];
  __shared__ DATATYPE bTile[block_K][block_N];
  if (row >= m || col >= n) return;

  ACCU_DATATYPE sum = 0.;
  for (int i = 0; i < k; i += block_K) {
    aTile[threadIdx.y][threadIdx.x] = a[row * k + threadIdx.x + i];
    bTile[threadIdx.y][threadIdx.x] = b[(i + threadIdx.y) * k + col];

    __syncthreads();
    for (int j = 0; j < block_K; j++) {
#if DATATYPE_BYTE == 4
      sum += aTile[threadIdx.y][j] * bTile[j][threadIdx.x];
#elif DATATYPE_BYTE == 2
      sum += __half2float(aTile[threadIdx.y][j] * bTile[j][threadIdx.x]);
#endif
    }
    __syncthreads();
  }
#if DATATYPE_BYTE == 4
  c[row * n + col] = sum;
#elif DATATYPE_BYTE == 2
  c[row * n + col] = __float2half(sum);
#endif
}

void matmul_gpu_naive_block_combine_access(DATATYPE *dev_a, DATATYPE *dev_b,
                                           DATATYPE *dev_c, int m, int n,
                                           int k) {
  uint3 grid = {m / block_M, n / block_N, 1};
  uint3 block = {block_M, block_N, 1};
  kernel_naive_block_combine_access<<<
      grid, block, (block_M + block_N) * sizeof(DATATYPE) * block_K>>>(
      dev_a, dev_b, dev_c, m, n, k);
}
