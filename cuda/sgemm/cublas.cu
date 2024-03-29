#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

using DATATYPE = float;

void cublas_matmul(cublasHandle_t& handle, DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m,
                      int n, int k) {
const float alpha = 1.0f;
const float beta = 0.0f;
//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, dev_b, n, dev_a, k, &beta, dev_c, n);
cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//              n, m, k, &alpha,  
//              dev_b, CUDA_R_32F, n, 
//              dev_a, CUDA_R_32F, k, 
//              &beta, 
//             dev_c, CUDA_R_32F, n, 
//             //CUBLAS_COMPUTE_32F_FAST_16F,
//             CUBLAS_COMPUTE_32F_FAST_TF32,
//             CUBLAS_GEMM_DFALT_TENSOR_OP);

// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
//                n, m, k, &alpha,
//                dev_b, CUDA_R_32F, n,
//                dev_a, CUDA_R_32F, k,
//                &beta,
//               dev_c, CUDA_R_32F, n,
//               CUBLAS_COMPUTE_32F_FAST_TF32,
//              CUBLAS_GEMM_DFALT_TENSOR_OP);

cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k, &alpha,
               dev_b, CUDA_R_32F, n,
               dev_a, CUDA_R_32F, k,
               &beta,
              dev_c, CUDA_R_32F, n);

}
