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

__global__ void kernel_naive0(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m,
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

// 每个cuda thread计算cuda_M * cuda_N 个结果呢！
#define cuda_M 4
#define cuda_N 4

__global__ void kernel_naive1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m,
                              int n, int k) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row = (idx / (n / cuda_N)) * cuda_M;
  const int col = (idx % (n / cuda_N)) * cuda_N;
  // row 和 col 是我这个cuda thread 需要处理的一小块！

  if (row >= m || col >= n) return;

  ACCU_DATATYPE sum[cuda_M][cuda_N] = {0.};
  DATATYPE reg_a[cuda_M];
  DATATYPE reg_b[cuda_N];

  for (int i = 0; i < k; i++) {
    for (int row_i = row; row_i < row + cuda_M; row_i++) {
      reg_a[row_i - row] = a[row_i * k + i];
    }
    for (int col_i = col; col_i < col + cuda_N; col_i++) {
      reg_b[col_i - col] = b[i * n + col_i];
    }

    for (int reg_i = 0; reg_i < cuda_M; reg_i++) {
      for (int reg_j = 0; reg_j < cuda_N; reg_j++) {
        sum[reg_i][reg_j] += reg_a[reg_i] * reg_b[reg_j];
      }
    }
  }

  for (int reg_i = 0; reg_i < cuda_M; reg_i++) {
    for (int reg_j = 0; reg_j < cuda_N; reg_j++) {
      c[(row + reg_i) * n + col + reg_j] = sum[reg_i][reg_j];
    }
  }
}

void matmul_gpu_naive(DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m,
                      int n, int k) {
  uint3 grid = {m * n / (32 * cuda_M * cuda_N) + 1, 1, 1};
  uint3 block = {32, 1, 1};
  kernel_naive1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
}
