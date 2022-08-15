#include <stdio.h>

/*
nvcc cutlass_vs_cublas.cu -o a.out -arch sm_75 -lcublas
-I/zhoukangkang/2022-04-28inference_try/cutlass/include/ && ./a.out && rm -rf
a.out
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
#if DATATYPE_BYTE == 4
    a[i] = (rand() % 9999) / 10000.0;
#else
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
#endif
  }
}

cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A,
                           int lda, float const *B, int ldb, float beta,
                           float *C, int ldc) {
  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible
  // compositions
  // including the following example for single-precision GEMM. Typical values
  // are used as
  // default template arguments. See
  // `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see
  // `cutlass/gemm/device/gemm.h`

  using ColumnMajor = cutlass::layout::ColumnMajor;

  using CutlassGemm =
      cutlass::gemm::device::Gemm<float,         // Data-type of A matrix
                                  ColumnMajor,   // Layout of A matrix
                                  float,         // Data-type of B matrix
                                  ColumnMajor,   // Layout of B matrix
                                  float,         // Data-type of C matrix
                                  ColumnMajor>;  // Layout of C matrix

  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that
  // are constructible
  // in host code and passed to kernels by value. These may include pointers,
  // strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for
  // passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel
  // entry.
  //
  CutlassGemm::Arguments args({M, N, K},  // Gemm Problem dimensions
                              {A, lda},   // Tensor-ref for source matrix A
                              {B, ldb},   // Tensor-ref for source matrix B
                              {C, ldc},   // Tensor-ref for source matrix C
                              {C, ldc},   // Tensor-ref for destination matrix D
                              // (may be different memory than source
                              // C matrix)
                              {alpha, beta});  // Scalars used in the Epilogue

  cutlass::Status status = gemm_operator(args);

  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
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
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n,
                dev_a, k, &beta, dev_c, n);

    // CutlassSgemmNN(n, m, k, alpha, dev_b, n, dev_a, k, beta, dev_c, n);
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
