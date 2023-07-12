#include "cublas_v2.h"
using DATATYPE = int8_t;
using BROADCAST_DATATYPE = int32_t;
using C_DATATYPE = int32_t;

void init(int8_t *a, int size);
void init(float *a, int size);
void init(int32_t *a, int size);

int32_t diff(const int32_t *c, const int32_t *c_baseline, int n);
int32_t diff(const int8_t *c, const int32_t *c_baseline, int n);
float diff(const float *c, const float *c_baseline, int n);



template <typename T>
void naive_gemm_cpu(const int8_t *a, const int8_t *b, T *c_cpu, int m,
                    int n, int k, const T * bias);

cudaError_t CutlassIgemmNN(int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B, int ldb, 
                           float const *bias,
                           C_DATATYPE *C, int ldc);
cudaError_t CutlassIgemmNN_sm80(int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B, int ldb, 
                           float const *bias,
                           C_DATATYPE *C, int ldc);
cudaError_t GemmWithBroadcast(int M, int N, int K, 
                              DATATYPE const *A, int lda, 
                              DATATYPE const *B, int ldb,
                              const BROADCAST_DATATYPE  *broadcast,
                              C_DATATYPE *C, int ldc);

void CublasIgemmNN(cublasHandle_t& handle, int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B,int ldb, 
                           int32_t const *bias,
                           C_DATATYPE *C, int ldc);
