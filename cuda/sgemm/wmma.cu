#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include <mma.h>
#include "cublas_v2.h"
using namespace nvcuda;

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
using ACCU_DATATYPE = float;
#define DATATYPE_BYTE 2
#define ACCU_DATATYPE_BYTE 4

// 每个 warp 计算 warp_M * warp_N 个结果
#define warp_M 16
#define warp_N 16
#define warp_K 16
#define WARP_SIZE 32

__global__ void matmul_gpu1(DATATYPE *a, DATATYPE *b, float *c, int m, int n,
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

void init(DATATYPE *a, int size) {
  for (int i = 0; i < size; i++) {
#if DATATYPE_BYTE == 4
    a[i] = (rand() % 9999) / 10000.0;
#else
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
#endif
  }
}

int main(void) {
  int m = 512;
  int n = 512;
  int k = 512;
  DATATYPE *a, *b;
  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  if (status != cudaSuccess) {
    printf("分配paged内存失败");
  }
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);
  if (status != cudaSuccess) {
    printf("分配paged内存失败");
  }
  init(a, m * k);
  init(b, k * n);

  ACCU_DATATYPE *c;
  cudaMallocHost(&c, sizeof(ACCU_DATATYPE) * m * n);
  memset(c, 0, sizeof(ACCU_DATATYPE) * m * n);

  float *c_cpu_fp32 = (float *)malloc(sizeof(float) * m * n);
  memset(c_cpu_fp32, 0, sizeof(float) * m * n);

  DATATYPE *dev_a, *dev_b;
  ACCU_DATATYPE *dev_c;
  cublasHandle_t handle;
  cublasCreate(&handle);

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
#if DATATYPE_BYTE == 4
        sum += a[i * k + ii] * b[ii * n + j];
#else
        sum += __half2float(a[i * k + ii]) * __half2float(b[ii * n + j]);
#endif
      }
      c_cpu_fp32[i * n + j] = sum;
    }
  }

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  double max_diff = -1.;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
#if ACCU_DATATYPE_BYTE == 4
      double c_gpu_fp32 = c[i * n + j];
#else
      double c_gpu_fp32 = __half2float(c[i * n + j]);
#endif
      if (std::abs(c_cpu_fp32[i * n + j] - c_gpu_fp32) > max_diff) {
        max_diff = std::abs(c_cpu_fp32[i * n + j] - c_gpu_fp32);
      }
    }
  }

  printf("max_diff: %f\n", max_diff);

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu_fp32);
  return 0;
}
