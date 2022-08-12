#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = float;
using ACCU_DATATYPE = float;
#define DATATYPE_BYTE 4
#define ACCU_DATATYPE_BYTE 4

// 每个 block 含有block_M * block_N 个结果
// 每个block计算（block_M * cuda_M） * （block_N * cuda_N）个结果呢！
#define block_M 16
#define block_N 16

#define block_K 4
// 每个cuda thread计算cuda_M * cuda_N 个结果呢！
#define cuda_M 4
#define cuda_N 4

__global__ void matmul_gpu1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m, int n,
                            int k) {
  const int row = (threadIdx.x + blockIdx.x * blockDim.x) * cuda_M;
  const int col = (threadIdx.y + blockIdx.y * blockDim.y) * cuda_N;
  const int row_in_block = (threadIdx.x) * cuda_M;
  const int col_in_block = (threadIdx.y) * cuda_N;

  __shared__ float aTile[block_K][block_M * cuda_M];
  __shared__ float bTile[block_K][block_N * cuda_N];
  float cTile[cuda_M][cuda_N] = {0};
  DATATYPE a_reg[4];
  DATATYPE b_reg[4];

  for (int i = 0; i < k; i += block_K) {
    // threadIdx.x 和 threadIdx.y都是[0-15]之间的数字
    // 每个block计算（16*4）* （16*4）这么多数字！，也就是block中每个cuda
    // thread计算4*4个数字
    int thread_id_in_block = threadIdx.x + threadIdx.y * block_M;
    // printf("%d\n", thread_id_in_block);
    int aTile_x = thread_id_in_block / cuda_N;
    int aTile_y = thread_id_in_block % cuda_N;
    aTile[aTile_y][aTile_x] =
        a[(aTile_x + blockIdx.x * block_N * cuda_N) * k + aTile_y + i];
    int bTile_x = thread_id_in_block / (block_N * cuda_N);
    int bTile_y = thread_id_in_block % (block_N * cuda_N);
    bTile[bTile_x][bTile_y] =
        b[(i + bTile_x) * n + blockIdx.y * block_N * cuda_N + bTile_y];

    __syncthreads();

    // 下面是计算一个16*4 * 4 与 4 * 16*4
    // 其实是 4 * 4 与 4 * 4 的乘积
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
          cTile[cTile_i][cTile_j] += a_reg[cTile_i] * b_reg[cTile_j];
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

  uint3 grid = {m / (block_M * cuda_M), n / (block_N * cuda_N), 1};
  uint3 block = {block_M, block_N, 1};

  for (int i = 0; i < WARMUP; i++) {
    // const float alpha=1.0f;
    // const float beta=0.0f;
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
    //                           n,m,k,
    //                           &alpha,
    //                           dev_b,n,
    //                           dev_a,k,
    //                           &beta,
    //                           dev_c,n);
    matmul_gpu1<<<grid, block,
                  (block_M * cuda_M + block_N * cuda_N) * sizeof(float) *
                      block_K>>>(dev_a, dev_b, dev_c, m, n, k);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    // const float alpha=1.0f;
    // const float beta=0.0f;
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
    //                           n,m,k,
    //                           &alpha,
    //                           dev_b,n,
    //                           dev_a,k,
    //                           &beta,
    //                           dev_c,n);
    matmul_gpu1<<<grid, block,
                  (block_M * cuda_M + block_N * cuda_N) * sizeof(float) *
                      block_K>>>(dev_a, dev_b, dev_c, m, n, k);
    // gemm_kernel_NN<<<grid, block>>>(dev_a, dev_b, (float4*)(dev_c), 1, 0, m,
    // n, k);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(beg);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("%f\n", elapsed_time);

  cudaMemcpy(c, dev_c, m * n * sizeof(ACCU_DATATYPE), cudaMemcpyDeviceToHost);

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

  printf("%f\n", max_diff);

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu_fp32);
  return 0;
}
