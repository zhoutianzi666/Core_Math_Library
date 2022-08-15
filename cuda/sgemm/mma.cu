#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"

using DATATYPE = half;
using ACCU_DATATYPE = float;

__global__ void matmul_gpu1(DATATYPE *a, DATATYPE *b, ACCU_DATATYPE *c, int m,
                            int n, int k) {
  float Accum[8];
  uint2 MultiA[1];
  uint2 MultiB[1];
  for (int i = 0; i < 8; ++i) {
    Accum[(i)] = 0.000000e+00f;
  }

  for (int i = 0; i < 1; i++)
  {
    int row_a = 0;
    if (threadIdx.x < 16)
        row_a = threadIdx.x % 4;
    else 
        row_a = threadIdx.x % 4 + 4;
    row_a += ((threadIdx.x / 4 ) % 2) * 8;

    MultiA[0] = ((uint2 *)((row_a * k) + a))[0];

    int b_col = 0;
    int b_row = threadIdx.x % 4;

    if (threadIdx.x / 8 == 0)
    {
        b_col = 0;
    }
    else if (threadIdx.x / 8 == 2) {
        b_col = 8;
    }
    else if(threadIdx.x / 8 == 1) {
        b_col = 4;
    }
    else {
        b_col = 12;
    }

    MultiB[0] = ((uint2 *)((b_row * n) + b_col + b))[0];


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
    c[(((((((((((((int)threadIdx.x) & 7) >> 2) * 128) +
              ((((int)threadIdx.x) >> 4) * 64)) +
             (((mma_accum_c_id & 3) >> 1) * 32)) +
            ((((int)threadIdx.x) & 1) * 16)) +
           ((mma_accum_c_id >> 2) * 8)) +
          (((((int)threadIdx.x) & 15) >> 3) * 4)) +
         (((((int)threadIdx.x) & 3) >> 1) * 2)) +
        (mma_accum_c_id & 1)))] = Accum[(mma_accum_c_id)];
  }
}

/*

__global__ void matmul_gpu2(DATATYPE *a, DATATYPE *b, ACCU_DATATYPE *c, int m,
                            int n, int k) {
  float Accum[8];
  uint2 MultiA[1];
  uint2 MultiB[1];
  for (int i = 0; i < 8; ++i) {
    Accum[(i)] = 0.000000e+00f;
  }

  for (int i = 0; i < 1; i++)
  {
    int row_a = 0;
    if (threadIdx.x < 16)
        row_a = threadIdx.x % 4;
    else 
        row_a = threadIdx.x % 4 + 4;
    row_a += (1 - (threadIdx.x / 4 ) % 2) * 8;

    MultiA[0] = ((uint2 *)((row_a * k) + a))[0];

    MultiB[0] = ;

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
    c[(((((((((((((int)threadIdx.x) & 7) >> 2) * 128) +
              ((((int)threadIdx.x) >> 4) * 64)) +
             (((mma_accum_c_id & 3) >> 1) * 32)) +
            ((((int)threadIdx.x) & 1) * 16)) +
           ((mma_accum_c_id >> 2) * 8)) +
          (((((int)threadIdx.x) & 15) >> 3) * 4)) +
         (((((int)threadIdx.x) & 3) >> 1) * 2)) +
        (mma_accum_c_id & 1)))] = Accum[(mma_accum_c_id)];
  }
}
*/

int main(void) {
  int m = 16;
  int n = 16;
  int k = 4;
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
  for (int i = 0; i < m * k; i++) a[i] = __float2half((rand() % 9999) / 1000.0);

  for (int i = 0; i < k * n; i++) b[i] = __float2half((rand() % 9999) / 1000.0);

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

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  uint3 grid = {1, 1, 1};
  uint3 block = {32, 1, 1};
  for (int i = 0; i < 100; i++) {
    matmul_gpu1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
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
      if (std::abs(c_cpu[i * n + j] - c[i * n + j]) > max_diff) {
        max_diff = std::abs(c_cpu[i * n + j] - c[i * n + j]);
      }
    }
  }

  printf("%f\n", max_diff);

  cudaDeviceReset();
  cudaFreeHost(a);
  cudaFreeHost(b);
  cudaFreeHost(c);
  free(c_cpu);
  return 0;
}