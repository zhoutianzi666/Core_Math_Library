#include <cuda_fp16.h>
void init(float *a, int size);
void init(half *a, int size);
void naive_gemm_cpu(const half *a, const half *b, float *c_cpu_fp32, int m,
                    int n, int k);
void naive_gemm_cpu(const float *a, const float *b, float *c_cpu_fp32, int m,
                    int n, int k);
float diff(const half *c, const float *c_baseline, int n);
float diff(const float *c, const float *c_baseline, int n);

void matmul_gpu(half *dev_a, half *dev_b, half *dev_c, int m, int n, int k);

void matmul_gpu_naive(float *dev_a, float *dev_b, float *dev_c, int m, int n,
                      int k);

void matmul_gpu_mma(half *dev_a, half *dev_b, float *dev_c, int m, int n,
                    int k);
void matmul_wmma(half *dev_a, half *dev_b, float *dev_c, int m, int n, int k);
void matmul_wmma(half *dev_a, half *dev_b, half *dev_c, int m, int n, int k);

void matmul_gpu_naive_block(float *dev_a, float *dev_b, float *dev_c, int m,
                            int n, int k);
void matmul_gpu_naive_block_combine_access(float *dev_a, float *dev_b,
                                           float *dev_c, int m, int n, int k);
cudaError_t CutlassHgemmNN(int M, int N, int K, half alpha, half const *A,
                           int lda, half const *B, int ldb, half beta, half *C,
                           int ldc);
