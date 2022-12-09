#include <cuda_fp16.h>
#include "cublas_v2.h"

void init(float *a, int size);
void naive_gemm_cpu(const float *a, const float *b, float *c_cpu_fp32, int m,
                    int n, int k);

float diff(const float *c, const float *c_baseline, int n);

void matmul_gpu(float *dev_a, float *dev_b, float *dev_c, int m, int n, int k);

void matmul_gpu_naive(float *dev_a, float *dev_b, float *dev_c, int m, int n,
                      int k);

void matmul_gpu_megengine(float *dev_a, float *dev_b, float *dev_c, int m,
                          int n, int k);

void matmul_gpu_naive_block(float *dev_a, float *dev_b, float *dev_c, int m,
                            int n, int k);
void matmul_gpu_naive_block_combine_access(float *dev_a, float *dev_b,
                                           float *dev_c, int m, int n, int k);
cudaError_t CutlassSgemmNN(int M, int N, int K, float alpha, float const *A,
                           int lda, float const *B, int ldb, float beta,
                           float *C, int ldc);
void cublas_matmul(cublasHandle_t& handle, float *dev_a, float *dev_b, float *dev_c, int m, int n, int k);
