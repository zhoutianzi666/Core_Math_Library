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

// it is always float
using ACCU_DATATYPE = float;

// 每个 block 含有block_M * block_N 个结果
// 每个block计算（block_M * cuda_M） * （block_N * cuda_N）个结果呢！
#define block_M 16
#define block_N 16

#define block_K 4
// 每个cuda thread计算cuda_M * cuda_N 个结果呢！
#define cuda_M 4
#define cuda_N 4

__global__ void kernel_gpu(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m, int n,
                           int k) {
  const int row = (threadIdx.x + blockIdx.x * blockDim.x) * cuda_M;
  const int col = (threadIdx.y + blockIdx.y * blockDim.y) * cuda_N;
  const int row_in_block = (threadIdx.x) * cuda_M;
  const int col_in_block = (threadIdx.y) * cuda_N;

  __shared__ DATATYPE aTile[block_K][block_M * cuda_M];
  __shared__ DATATYPE bTile[block_K][block_N * cuda_N];
  ACCU_DATATYPE cTile[cuda_M][cuda_N] = {0};
  DATATYPE a_reg[cuda_M];
  DATATYPE b_reg[cuda_N];

  for (int i = 0; i < k; i += block_K) {
    // threadIdx.x 和 threadIdx.y都是[0-15]之间的数字
    // 每个block计算（block_M* cuda_M）* （block_N * cuda_N）这么多数字！，
    // 也就是block中每个cuda thread计算cuda_M * cuda_N个数字
    int thread_id_in_block = threadIdx.x + threadIdx.y * block_M;
    // printf("%d\n", thread_id_in_block);
    int aTile_x = thread_id_in_block / block_K;
    int aTile_y = thread_id_in_block % block_K;
    aTile[aTile_y][aTile_x] =
        a[(aTile_x + blockIdx.x * block_N * cuda_N) * k + aTile_y + i];
    int bTile_x = thread_id_in_block / (block_N * cuda_N);
    int bTile_y = thread_id_in_block % (block_N * cuda_N);
    bTile[bTile_x][bTile_y] =
        b[(i + bTile_x) * n + blockIdx.y * block_N * cuda_N + bTile_y];

    __syncthreads();

    // 下面是计算一个（block_M* cuda_M） * block_K 与 block_K * （block_N*
    // cuda_N） 其实是 4 * 4 与 4 * 4 的乘积
    for (int j = 0; j < block_K; j++) {
      for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
        a_reg[cTile_i] = aTile[j][row_in_block + cTile_i];
      }

      for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
        b_reg[cTile_j] = bTile[j][col_in_block + cTile_j];
      }
#pragma unroll
      for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
        for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
#if DATATYPE_BYTE == 4
          cTile[cTile_i][cTile_j] += a_reg[cTile_i] * b_reg[cTile_j];
#elif DATATYPE_BYTE == 2
          cTile[cTile_i][cTile_j] +=
              __half2float(a_reg[cTile_i] * b_reg[cTile_j]);
#endif
        }
      }
    }
    __syncthreads();
  }

  for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
    for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
      c[(row + cTile_i) * n + col + cTile_j] = cTile[cTile_i][cTile_j];
    }
  }
}

void matmul_gpu(DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m, int n,
                int k) {
  uint3 grid = {m / (block_M * cuda_M), n / (block_N * cuda_N), 1};
  uint3 block = {block_M, block_N, 1};
  kernel_gpu<<<grid, block,
               (block_M * cuda_M + block_N * cuda_N) * sizeof(float) *
                   block_K>>>(dev_a, dev_b, dev_c, m, n, k);
}
