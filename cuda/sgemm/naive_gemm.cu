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

__global__ void kernel_naive(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m,
                             int n, int k) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = idx / n;
  const int col = idx % n;

  if (row >= m || col >= n) return;

  ACCU_DATATYPE sum = 0.;
  for (int i = 0; i < k; i++) {
#if DATATYPE_BYTE == 4
    sum += a[row * k + i] * b[i * n + col];
#elif DATATYPE_BYTE == 2
    sum += __half2float(a[row * k + i] * b[i * n + col]);
#endif
  }

#if DATATYPE_BYTE == 4
  c[row * n + col] = sum;
#elif DATATYPE_BYTE == 2
  c[row * n + col] = __float2half(sum);
#endif
}

#define block_K 512
__global__ void matmul_gpu2(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m, int n,
                            int k) {
  const int tidx = threadIdx.x;
  const int bidx = blockIdx.x;
  int idx = tidx + bidx * blockDim.x;
  const int row = idx / n;
  const int col = idx % n;
  __shared__ DATATYPE aTile[block_K];

  if (row >= m || col >= n) return;

  ACCU_DATATYPE sum = 0.;

  for (int i = 0; i < k; i += block_K) {
    if (tidx < block_K && tidx + i < k) {
      aTile[tidx] = a[row * k + tidx + i];
    }

    __syncthreads();

    for (int j = i; j < i + block_K; j++) {
#if DATATYPE_BYTE == 4
      sum += aTile[j - i] * b[j * n + col];
#elif DATATYPE_BYTE == 2
      sum += __half2float(aTile[j - i] * b[j * n + col]);
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

void matmul_gpu_naive(DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m,
                      int n, int k) {
  uint3 grid = {m * n / 512 + 1, 1, 1};
  uint3 block = {512, 1, 1};
  kernel_naive<<<grid, block, 0 * block_K * sizeof(DATATYPE)>>>(dev_a, dev_b,
                                                                dev_c, m, n, k);
}
