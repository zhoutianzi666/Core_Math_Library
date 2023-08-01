#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm_template.h"

#include "fpA_intB_gemm.h"

#include <stdio.h>

#include <cuda_fp16.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#define WARMUP 0
#define REPEATE 1

using DATATYPE = half;
using B_DATATYPE = int8_t;
using scale_DATATYPE = float;

void CUDA_CHECK(cudaError_t status) {
  if (status != cudaSuccess) {
    printf("分配paged内存失败\n");
  }
}

void init(int8_t *a, int size) {
    for (int i = 0; i < size; i++) {
      a[i] = rand() % 256 - 128;
    }
}
  
void init(half *a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = (half)(rand() % 256 / 2560.f);
    }
}

void init(float *a, int size) {
    for (int i = 0; i < size; i++) {
        a[i] = 0.1;
    }
}


int main(void) {
  int m = 85;
  int n = 15360;
  int k = 5120;

  // a,b,c is in cpu place!
  DATATYPE *a;
  B_DATATYPE *b, *origin_b;
  scale_DATATYPE *scale;
  DATATYPE *c;

  cudaError_t status = cudaMallocHost(&a, sizeof(DATATYPE) * m * k);
  CUDA_CHECK(status);
  status = cudaMallocHost(&b, sizeof(B_DATATYPE) * k * n);
  CUDA_CHECK(status);
  status = cudaMallocHost(&origin_b, sizeof(B_DATATYPE) * k * n);
  CUDA_CHECK(status);
  status = cudaMallocHost(&c, sizeof(DATATYPE) * m * n);
  CUDA_CHECK(status);
  memset(c, 0, sizeof(DATATYPE) * m * n);
  status = cudaMallocHost(&scale, sizeof(scale_DATATYPE) * n);
  CUDA_CHECK(status);

  init(a, m * k);
  init(b, k * n);
  memcpy(origin_b, b, k * n);
  
  // 将b更新一下，用来调用cutlass！
  for(int i =0 ; i < k *n;i++) {
    b[i] = (b[i] + 128);
  }

  for(int i =0 ; i < k *n;i += 4) {
    auto tmp = b[i + 1];
    b[i + 1] = b[i + 2];
    b[i + 2] = tmp;
  }

  init(c, m * n);
  init(scale, n);

  DATATYPE *dev_a;
  B_DATATYPE *dev_b;
  scale_DATATYPE *dev_scale;
  DATATYPE *dev_c;

  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  // allocate the memory on the GPU and copy a and b to GPU
  cudaMalloc((void **)&dev_a, m * k * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_b, k * n * sizeof(B_DATATYPE));
  cudaMalloc((void **)&dev_scale, n * sizeof(scale_DATATYPE));
  cudaMalloc((void **)&dev_c, m * n * sizeof(DATATYPE));
  cudaMemcpy(dev_a, a, m * k * sizeof(DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, k * n * sizeof(B_DATATYPE), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_scale, scale, n * sizeof(scale_DATATYPE), cudaMemcpyHostToDevice);

  cudaEvent_t beg, end;
  auto int8_mixed_gemm_runner = CutlassFpAIntBGemmRunner<half, uint8_t>();

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  for (int i = 0; i < WARMUP + REPEATE; i++) {
    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }
    int8_mixed_gemm_runner.gemm(   dev_a,
                                   (const uint8_t*)dev_b,
                                   dev_scale,
                                   dev_c,
                                   m,n,k,
                                   nullptr,0,stream);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu gemm compute time: %f\n", elapsed_time);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);


half *c_cpu_fp16 = (half *)malloc(sizeof(half) * m * n);
half *c_from_gpu = (half *)malloc(sizeof(half) * m * n);

cudaMemcpy(c_from_gpu, dev_c, m * n * sizeof(half), cudaMemcpyDeviceToHost);


float max_diff = -10;


for(int ii = 62; ii < m;ii+=100) {
    for (int jj = 0; jj < n; jj++) {
        float sum = 0.f;
        for(int kk = 0; kk < k; kk++) {
            int to_address = kk * n + jj;
            sum +=  scale[jj] * (int)(origin_b[to_address]) * (float)(a[ii * k + kk]);
        }
        c_cpu_fp16[ii * n + jj] = (half)(sum);
        float tmp =  (float)(c_from_gpu[ii * n + jj]);
        if (std::abs(tmp - sum) > max_diff)
        {
            max_diff = std::abs(tmp - sum);
            std::cout << max_diff << std::endl;
            std::cout << tmp << " " <<  sum << std::endl;
            std::cout << ii << " " <<  jj << std::endl;
        }
    }
}

printf("max_diff : %f\n", max_diff);

  cudaFreeHost(dev_a);
  cudaFreeHost(dev_b);
  cudaFreeHost(dev_scale);
  cudaFreeHost(dev_c);
  free(c_cpu_fp16);
  free(c_from_gpu);
  cudaDeviceReset();
  return 0;
}


