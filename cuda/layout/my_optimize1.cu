#include "utility.h"
#include <stdio.h>
#include "assert.h" 

using DATATYPE = half;

using DATATYPE2 = half2;

template <int32_t accessSize = 8, typename accessType = float,
int blockM = 16, int blockN = 16, int padding = 2>
__global__ void my_row_col_kernel1(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  
  __shared__ DATATYPE aTile[blockM * (blockN * accessSize + padding)];
  int smem_row = blockM;
  int smem_col = blockN * accessSize + padding;
  int vol = M * N;
  int batch_i = blockIdx.z;

  // g_row_0 , g_col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是blockM和blockN*accessSize的倍数！！
  int g_row_0 = blockIdx.x * blockDim.x;
  int g_col_0 = blockIdx.y * blockDim.y * accessSize;

  int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  // local column and row read by the current cuda thread
  int local_rowi = thread_id_in_block / blockN;
  int local_coli = thread_id_in_block % blockN;

  int g_row_i = g_row_0 + local_rowi;
  int g_col_i = g_col_0 + local_coli * accessSize;

  int input_offset = batch_i * vol + g_row_i * N + g_col_i; 
  DATATYPE tmp[accessSize];
  if (g_row_i < M && g_col_i < N) 
  {
    *(float4*)(tmp) = *(float4*)(input + input_offset);
    for (int i = 0 ; i < accessSize; i++)
    {
     aTile[local_rowi * smem_col + local_coli * accessSize + i] = tmp[i];
     //aTile[(local_coli * accessSize) * smem_col + local_rowi + i * smem_col] = tmp[i];
    }
  }

  __syncthreads();

  g_row_i = g_col_0 + local_rowi;
  g_col_i = g_row_0 + local_coli * accessSize;
  
  int output_offset = batch_i * vol + g_row_i * M + g_col_i;
  //int output_offset = batch_i * vol + g_row_i * N + g_col_i; 
  
  for (int i = 0; i < accessSize; i++) {
    tmp[i] = aTile[(local_coli * accessSize + i) * smem_col + local_rowi];
  }

  *(float4*)(output + output_offset) = *(float4*)(tmp);
}

/*
思路是：将输入分成三个部分，M，N， batch这三个部分
每个thread block计算 blockM * (blockN * accessSize) 这么大部分！
每个thread block的线程维度是(blockM, blockN)
对于每个batch，其实就是将行矩阵存储变成列存储矩阵！
*/




template <int32_t accessSize = 8, typename accessType = float,
int blockM = 16, int blockN = 16, int padding = 2>
__global__ void my_row_col_kernel2(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  
  __shared__ DATATYPE aTile[blockM * (blockN * accessSize + padding)];
  int smem_row = blockM;
  int smem_col = blockN * accessSize + padding;
  int vol = M * N;
  int batch_i = blockIdx.z;

  // g_row_0 , g_col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是blockM和blockN*accessSize的倍数！！
  int g_row_0 = blockIdx.x * blockDim.x;
  int g_col_0 = blockIdx.y * blockDim.y * accessSize;

  int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  // local column and row read by the current cuda thread
  int local_rowi = thread_id_in_block / blockN;
  int local_coli = thread_id_in_block % blockN;

  int g_row_i = g_row_0 + local_rowi;
  int g_col_i = g_col_0 + local_coli * accessSize;

  int input_offset = batch_i * vol + g_row_i * N + g_col_i; 
  DATATYPE tmp[accessSize];
  if (g_row_i < M && g_col_i < N) 
  {
    *(float4*)(tmp) = *(float4*)(input + input_offset);
    for (int i = 0 ; i < accessSize; i++)
    {
     aTile[local_rowi * smem_col + local_coli * accessSize + i] = tmp[i];
     //aTile[(local_coli * accessSize) * smem_col + local_rowi + i * smem_col] = tmp[i];
    }
  }

  __syncthreads();

  g_row_i = g_col_0 + local_rowi;
  g_col_i = g_row_0 + local_coli;
  
  for (int i = 0; i < accessSize; i++) {
    int output_offset = batch_i * vol + g_row_i * M + g_col_i;
    half tmp_v = aTile[local_coli * smem_col + local_rowi];
    *(output + output_offset) = tmp_v; 
    g_col_i  += accessSize;
    local_coli += accessSize;
  }
}


// 下面的代码还在行的维度上做条带挖掘！，牛逼楼！

template <int32_t accessSize = 8, typename accessType = float,
int blockM = 16, int blockN = 16, int padding = 2,
int tileM = 16, int tileN = blockN * accessSize
>
__global__ void my_row_col_kernel3(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  
  __shared__ half aTile[tileM][tileN + padding];
  int vol = M * N;
  int batch_i = blockIdx.z;

  // g_row_0 , g_col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是tileM和tileN的倍数！！
  int g_row_0 = blockIdx.x * tileM;
  int g_col_0 = blockIdx.y * tileN;

  int tid_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  // local column and row read by the current cuda thread
  int local_row_start = tid_in_block / blockN;
  int row_step = blockM;
  int local_col = tid_in_block % blockN;
  
  for (int ri = local_row_start; ri < tileM; ri += row_step)
  {
    int g_row_i = g_row_0 + ri;
    int g_col_i = g_col_0 + local_col * accessSize;
    int input_offset = batch_i * vol + g_row_i * N + g_col_i; 
    if (g_row_i < M && g_col_i < N) 
    { 
        half tmp[accessSize];
        *(accessType*)(tmp) = *(accessType*)(input + input_offset);
        for (int i = 0 ; i < accessSize; i++)
        {
          aTile[ri][local_col * accessSize + i] = tmp[i];
        }
    }
  }

  __syncthreads();
  
  for (int i =0; i < accessSize * (tileM / blockM); i++) {

    int new_idx = (tid_in_block + i * blockM * blockN) % tileM;
    int new_idy = (tid_in_block + i * blockM * blockN) / tileM;
    int g_row_i = g_col_0 + new_idy;
    int g_col_i = g_row_0 + new_idx;

    if(g_row_i >= N || g_col_i >= M) break;
    int output_offset = batch_i * vol + g_row_i * M + g_col_i;
    *(output + output_offset ) = aTile[new_idx][new_idy];
  }
}


void my_row_col1(DATATYPE *output, const DATATYPE *input, int batch,
                           int M, int N) {

  if (N % 8 == 0 ) {
  const int blockM = 16;
  const int blockN = 16;
  const int accessSize = 8;
  const int tileM = 16;
  const int tileN = blockN * accessSize;
  const int padding = 2;
  assert(N % accessSize == 0);
  uint3 grid = {(M + tileM - 1) / tileM, (N + tileN - 1) / tileN, batch};
  uint3 block = {blockM, blockN, 1};
  my_row_col_kernel3<accessSize, float4, blockM, blockN, padding, tileM, tileN><<<grid, block>>>(output, input, batch, M, N);
  } else if (N % 4 == 0 ) {
  const int blockM = 16;
  const int blockN = 32;
  const int accessSize = 4;
  const int tileM = 32;
  const int tileN = 32 * accessSize;
  const int padding = 2;
  assert(N % accessSize == 0);
  uint3 grid = {(M + tileM - 1) / tileM, (N + tileN - 1) / (tileN), batch};
  uint3 block = {blockM, blockN, 1};
  my_row_col_kernel3<accessSize, float2, blockM, blockN, padding, tileM, tileN><<<grid, block>>>(output, input, batch, M, N);
  } else if (N % 2 == 0 ) {
  const int blockM = 8;
  const int blockN = 64;
  const int accessSize = 2;
  const int tileM = 32;
  const int tileN = blockN * accessSize;
  const int padding = 2;
  assert(N % accessSize == 0);
  uint3 grid = {(M + tileM - 1) / tileM, (N + tileN - 1) / (tileN), batch};
  uint3 block = {blockM, blockN, 1};
  my_row_col_kernel3<accessSize, half2, blockM, blockN, padding, tileM, tileN><<<grid, block>>>(output, input, batch, M, N);
  } else if (1) {
  const int blockM = 4;
  const int blockN = 128;
  const int accessSize = 1;
  const int tileM = 32;
  const int tileN = blockN * accessSize;
  const int padding = 2;
  assert(N % accessSize == 0);
  uint3 grid = {(M + tileM - 1) / tileM, (N + tileN - 1) / (tileN), batch};
  uint3 block = {blockM, blockN, 1};
  my_row_col_kernel3<accessSize, half, blockM, blockN, padding, tileM, tileN><<<grid, block>>>(output, input, batch, M, N);
  }

}
