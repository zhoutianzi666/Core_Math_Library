#include <algorithm>
#include <iostream>

#include "cublas_v2.h"
#include "utility.h"



void CublasIgemmNN(cublasHandle_t& handle, int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B,int ldb, 
                           int32_t const *bias,
                           C_DATATYPE *C, int ldc) {

const int alpha = 1;
const int beta = 0;

cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
               N, M, K, &alpha,
               B, CUDA_R_8I, N,
               A, CUDA_R_8I, K,
               &beta,
              C, CUDA_R_32I, N,
              CUBLAS_COMPUTE_32I,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

