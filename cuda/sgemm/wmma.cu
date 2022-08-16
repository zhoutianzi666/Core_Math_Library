#include <mma.h>
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

using namespace nvcuda;

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
#define DATATYPE_BYTE 2

using ACCU_DATATYPE = float;

// 每个 warp 计算 warp_M * warp_N 个结果
#define warp_M 16
#define warp_N 16
#define warp_K 16
#define WARP_SIZE 32

__global__ void kernel_wmma(DATATYPE *a, DATATYPE *b, float *c, int m, int n,
                            int k) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = idx / WARP_SIZE;
  // 每个warp计算warp_M * warp_N的结果

  int m_tile_num = m / warp_M;
  int n_tile_num = n / warp_N;

  int m_tile_id = warp_id / n_tile_num;
  int n_tile_id = warp_id % n_tile_num;

  wmma::fragment<wmma::matrix_a, warp_M, warp_N, warp_K, DATATYPE,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, warp_M, warp_N, warp_K, DATATYPE,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, warp_M, warp_N, warp_K, float> c_frag;

  float *c_unique = c + m_tile_id * warp_M * n + n_tile_id * warp_N;

  // Initialize the output to zero
  wmma::fill_fragment(c_frag, 0.0f);

  for (int i = 0; i < k; i += warp_K) {
    DATATYPE *a_unique = a + i + m_tile_id * warp_M * k;
    DATATYPE *b_unique = b + i * n + n_tile_id * warp_N;
    wmma::load_matrix_sync(a_frag, a_unique, k);
    wmma::load_matrix_sync(b_frag, b_unique, n);
    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  wmma::store_matrix_sync(c_unique, c_frag, n, wmma::mem_row_major);
}

void matmul_wmma(DATATYPE *dev_a, DATATYPE *dev_b, float *dev_c, int m, int n,
                 int k) {
  uint3 grid = {m * n / (warp_M * warp_N * (512 / 32)), 1, 1};
  uint3 block = {512, 1, 1};
  kernel_wmma<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
}
