#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"

using DATATYPE = half;
using ACCU_DATATYPE = float;

#define WARMUP 10
#define REPEATE 10

void init(DATATYPE *a, int size) {
  for (int i = 0; i < size; i++) {
#if DATATYPE_BYTE == 4
    a[i] = (rand() % 9999) / 10000.0;
#else
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
#endif
  }
}

// 每个 warp 计算 warp_M * warp_N 个结果
#define warp_M 16
#define warp_N 16
#define warp_K 4
#define WARP_SIZE 32

__global__ void matmul_gpu1(DATATYPE *a, DATATYPE *b, float *c, int m, int n,
                            int k) {
  float Accum[8];
  uint2 MultiA[1];
  uint2 MultiB[1];
  for (int i = 0; i < 8; ++i) {
    Accum[(i)] = 0.000000e+00f;
  }

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = idx / WARP_SIZE;
  // 每个warp计算warp_M * warp_N的结果
  int m_tile_num = m / warp_M;
  int n_tile_num = n / warp_N;
  int lane_id = threadIdx.x % WARP_SIZE;

  int m_tile_id = warp_id / n_tile_num;
  int n_tile_id = warp_id % n_tile_num;

  c += (m_tile_id * warp_M * n + n_tile_id * warp_N);
  a += (m_tile_id * warp_M * k);
  b += (n_tile_id * warp_N);

  for (int i = 0; i < k; i += warp_K) {
    int a_row = 0;
    if (lane_id < 16)
      a_row = lane_id % 4;
    else
      a_row = lane_id % 4 + 4;
    a_row += ((lane_id / 4) % 2) * 8;
    int a_col = 0;
    MultiA[0] = ((uint2 *)(a + a_row * k + a_col + i))[0];

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

int main(void) {
  int m = 512;
  int n = 512;
  int k = 512;
  DATATYPE *a;
  DATATYPE *b;

  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);

  if (status != cudaSuccess) {
    printf("分配内存失败");
  }
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);

  if (status != cudaSuccess) {
    printf("分配内存失败");
  }
  // srand((unsigned)time(NULL));
  init(a, m * k);
  init(b, k * n);

  ACCU_DATATYPE *c;
  cudaMallocHost(&c, sizeof(ACCU_DATATYPE) * m * n);
  memset(c, 0, sizeof(ACCU_DATATYPE) * m * n);

  ACCU_DATATYPE *c_cpu = (ACCU_DATATYPE *)malloc(sizeof(ACCU_DATATYPE) * m * n);
  memset(c_cpu, 0, sizeof(ACCU_DATATYPE) * m * n);

  DATATYPE *dev_a, *dev_b;
  ACCU_DATATYPE *dev_c;

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(ACCU_DATATYPE));

  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  uint3 grid = {m * n / (warp_M * warp_N * (512 / 32)), 1, 1};
  uint3 block = {512, 1, 1};

  for (int i = 0; i < WARMUP; i++) {
    matmul_gpu1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    matmul_gpu1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu gemm compute time: %f\n", elapsed_time);

  cudaMemcpy(c, dev_c, m * n * sizeof(ACCU_DATATYPE), cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.f;
      for (int ii = 0; ii < k; ii++) {
        //  sum += a[i * k + ii] * b[ii * n + j];
        sum += __half2float(a[i * k + ii]) * __half2float(b[ii * n + j]);
      }
      c_cpu[i * n + j] = sum;
    }
  }
  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  double max_diff = -1;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      // printf("%f\n", c_cpu[i * n + j]);
      // printf("%f\n", c[i * n + j]);
      if (std::abs(c_cpu[i * n + j] - c[i * n + j]) > max_diff) {
        max_diff = std::abs(c_cpu[i * n + j] - c[i * n + j]);
      }
    }
  }

  printf("max_diff: %f\n", max_diff);

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu);
  return 0;
}