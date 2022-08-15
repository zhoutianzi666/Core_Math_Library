#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
#define DATATYPE_BYTE 2

using ACCU_DATATYPE = float;

#define block_M 32
#define block_N 32
#define block_K 32
__global__ void matmul_gpu3(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m, int n,
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

int main(void) {
  int m = 512;
  int n = 512;
  int k = 512;
  DATATYPE *a, *b;
  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  if (status != cudaSuccess) {
    printf("分配内存失败");
  }
  status = cudaMallocHost(&b, sizeof(DATATYPE) * k * n);
  if (status != cudaSuccess) {
    printf("分配内存失败");
  }
  for (int i = 0; i < m * k; i++) {
#if DATATYPE_BYTE == 4
    a[i] = (rand() % 9999) / 10000.0;
#else
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
#endif
  }
  for (int i = 0; i < k * n; i++) {
#if DATATYPE_BYTE == 4
    b[i] = (rand() % 9999) / 10000.0;
#else
    b[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
#endif
  }

  DATATYPE *c;
  cudaMallocHost(&c, sizeof(DATATYPE) * m * n);
  memset(c, 0, sizeof(DATATYPE) * m * n);

  float *c_cpu_fp32 = (float *)malloc(sizeof(float) * m * n);
  memset(c_cpu_fp32, 0, sizeof(float) * m * n);

  DATATYPE *dev_a, *dev_b;
  DATATYPE *dev_c;

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(DATATYPE));

  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(DATATYPE), cudaMemcpyHostToDevice);

  uint3 grid = {m / block_M, n / block_N, 1};
  uint3 block = {block_M, block_N, 1};

  for (int i = 0; i < WARMUP; i++) {
    matmul_gpu3<<<grid, block,
                  (block_M + block_N) * sizeof(DATATYPE) * block_K>>>(
        dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    matmul_gpu3<<<grid, block,
                  (block_M + block_N) * sizeof(DATATYPE) * block_K>>>(
        dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("%f\n", elapsed_time);

  cudaMemcpy(c, dev_c, m * n * sizeof(DATATYPE), cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu time:%lf\n", double(time2 - time1) * 1000);

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
#if DATATYPE_BYTE == 4
      double c_gpu_fp32 = c[i * n + j];
#else
      double c_gpu_fp32 = __half2float(c[i * n + j]);
#endif
      if (std::abs(c_cpu_fp32[i * n + j] - c_gpu_fp32) > max_diff) {
        max_diff = std::abs(c_cpu_fp32[i * n + j] - c_gpu_fp32);
      }
    }
  }

  printf("%f\n", max_diff);

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu_fp32);
  return 0;
}
