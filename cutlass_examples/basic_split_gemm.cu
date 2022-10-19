#include <stdio.h>

/*
nvcc basic_split_gemm.cu -o a.out -arch sm_75 -lcublas -I/zhoukangkang/2022-04-28inference_try/cutlass/include/ && ./a.out && rm -rf a.out
*/

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = float;
using ACCU_DATATYPE = float;
#define DATATYPE_BYTE 4
#define ACCU_DATATYPE_BYTE 4

void init(DATATYPE *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (rand() % 9999) / 10000.0;
  }
}

cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A,
                           int lda, float const *B, int ldb, float beta,
                           float *C, int ldc) {

  using ColumnMajor = cutlass::layout::ColumnMajor;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
  float,
  1,
  float,
  float>;
  using CutlassGemm = cutlass::gemm::device::Gemm<float,       
                                                  ColumnMajor,
                                                  float,
                                                  ColumnMajor,
                                                  float,
                                                  ColumnMajor,
                                                  float,
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm75,
                                                  cutlass::gemm::GemmShape<128,128, 8>,
                                                  cutlass::gemm::GemmShape<32,64, 8>,
                                                  cutlass::gemm::GemmShape<1,1, 1>, 
                                                  EpilogueOutputOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                                  2,
                                                  1,
                                                  1,
                                                  false >; // // 如果支持split-k，那么设置为True！
  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({M, N, K},
                              {A, lda},  
                              {B, ldb}, 
                              {C, ldc}, 
                              {C, ldc}, 
                              {alpha, beta});
  // 如果支持split-k，那么就将下面注释打开哦！
  // size_t bytes = CutlassGemm::get_workspace_size(args);
  // void * workspace;
  // cudaMalloc((void**)&workspace, bytes);
  cutlass::Status status = gemm_operator(args); //workspace);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
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

  for (int i = 0; i < WARMUP; i++) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
    //                           n,m,k,
    //                           &alpha,
    //                           dev_b,n,
    //                           dev_a,k,
    //                           &beta,
    //                           dev_c,n);

    CutlassSgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,
    //                           n,m,k,
    //                           &alpha,
    //                           dev_b,n,
    //                           dev_a,k,
    //                           &beta,
    //                           dev_c,n);

    CutlassSgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
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
        sum += a[i * k + ii] * b[ii * n + j];
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
      double c_gpu_fp32 = c[i * n + j];
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
