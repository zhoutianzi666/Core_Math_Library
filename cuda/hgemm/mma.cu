#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

using DATATYPE = half;
using ACCU_DATATYPE = float;

#define WARMUP 10
#define REPEATE 10

// 每个 warp 计算 warp_M * warp_N 个结果
#define warp_M 16
#define warp_N 16
#define warp_K 4
#define WARP_SIZE 32

__global__ void kernel_mma(DATATYPE *a, DATATYPE *b, float *c, int m, int n,
                           int k) {
  float Accum[8];
  uint2 MultiA[1];
  uint2 MultiB[1];
  for (int i = 0; i < 8; ++i) {
    Accum[(i)] = 0.000000e+00f;
  }

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = idx / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  // 每个warp计算warp_M * warp_N的结果
  int m_tile_num = m / warp_M;
  int n_tile_num = n / warp_N;

  int m_tile_id = warp_id / n_tile_num;
  int n_tile_id = warp_id % n_tile_num;

  c += (m_tile_id * warp_M * n + n_tile_id * warp_N);
  a += (m_tile_id * warp_M * k);
  b += (n_tile_id * warp_N);

  int a_row = 0;
  int a_col = 0;
  if (lane_id < 16)
    a_row = lane_id % 4;
  else
    a_row = lane_id % 4 + 4;
  a_row += ((lane_id / 4) % 2) * 8;

  int b_col = 0;
  int b_row = lane_id % 4;

  if (lane_id / 8 == 0) {
    b_col = 0;
  } else if (lane_id / 8 == 2) {
    b_col = 8;
  } else if (lane_id / 8 == 1) {
    b_col = 4;
  } else {
    b_col = 12;
  }

  for (int i = 0; i < k; i += warp_K) {
    MultiA[0] = ((uint2 *)(a + a_row * k + a_col + i))[0];
    MultiB[0] = ((uint2 *)((b_row * n) + b_col + b + i * n))[0];

    {
      unsigned const *A = reinterpret_cast<unsigned const *>(MultiA);
      unsigned const *B = reinterpret_cast<unsigned const *>(MultiB);
      float const *C = reinterpret_cast<float const *>(Accum);
      float *D = reinterpret_cast<float *>(Accum);
      __asm__ __volatile__(
          "mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
          "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, "
          "{%10,%11}, "
          "{%12,%13,%14,%15,%16,%17,%18,%19};\n"
          : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3]), "=f"(D[4]),
            "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
          : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]),
            "f"(C[2]), "f"(C[3]), "f"(C[4]), "f"(C[5]), "f"(C[6]), "f"(C[7]));
    }
  }

  for (int mma_accum_c_id = 0; mma_accum_c_id < 8; ++mma_accum_c_id) {
    /*
    下面这个是官网上的图示，结果他有错误！
    int c_row = 0;
    int X = (lane_id & 0b1) + (mma_accum_c_id & 0b10);
    c_row = X;
    if (lane_id >= 16)
    c_row = X + 4;
    int c_col = (mma_accum_c_id & 0b100) + (lane_id & 0b10) +
    (mma_accum_c_id & 0b1);
    int c_col_ = (lane_id / 4) % 2 * 8;
    int c_row_ =  ((lane_id / 4) % 4) >= 2 ? 8 : 0;
*/
    int c_row = (lane_id % 2) + ((mma_accum_c_id / 2 % 2) * 2) +
                4 * (lane_id / 16) + (lane_id % 16 / 4) % 2 * 8;
    int c_col = lane_id % 4 / 2 * 2 + lane_id % 16 / 8 * 4 +
                mma_accum_c_id % 2 + mma_accum_c_id / 4 * 8;
    c[c_row * n + c_col] = Accum[mma_accum_c_id];
  }
}

void matmul_gpu_mma(DATATYPE *dev_a, DATATYPE *dev_b, float *dev_c, int m,
                    int n, int k) {
  uint3 grid = {m * n / (warp_M * warp_N * (512 / 32)), 1, 1};
  uint3 block = {512, 1, 1};
  kernel_mma<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
}
