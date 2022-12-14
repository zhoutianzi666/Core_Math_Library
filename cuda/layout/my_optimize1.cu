#include "utility.h"
#include <stdio.h>
#include "assert.h" 

using DATATYPE = half;
#define blockM 32
#define blockN 32
#define padding 2
#define accessSize 8

__global__ void my_row_col_kernel1(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  
  __shared__ half aTile[blockM][blockN * accessSize + padding];
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
  if (g_row_i < M && g_col_i < N) 
  {
    half tmp[accessSize];
    *(float4*)(tmp) = *(float4*)(input + input_offset);
    for (int i = 0 ; i < accessSize; i++)
    {
      aTile[local_rowi][local_coli * accessSize + i] = tmp[i];
    }
  }

  __syncthreads();
  
  // 将aTile_tmp看成这个矩阵blockM * (blockN * accessSize)
  half* aTile_tmp = (half*)(aTile);
  int a_row = blockM;
  int a_col = blockN * accessSize + padding;


  for (int i =0; i < accessSize; i++) {

    int new_idx = (thread_id_in_block + i * blockDim.x * blockDim.y) % blockM;
    int new_idy = (thread_id_in_block + i * blockDim.x * blockDim.y) / blockM;
    g_row_i = g_col_0 + new_idy;
    g_col_i = g_row_0 + new_idx;

    if(g_row_i >= N || g_col_i >= M) break;
    int output_offset = batch_i * vol + g_row_i * M + g_col_i;
    *(output + output_offset ) = aTile_tmp[new_idx * a_col + new_idy];
  }
}

/*
思路是：将输入分成三个部分，M，N， batch这三个部分
每个thread block计算 blockM * (blockN * accessSize) 这么大部分！
每个thread block的线程维度是(blockM, blockN)
对于每个batch，其实就是将行矩阵存储变成列存储矩阵！
*/

void my_row_col1(DATATYPE *output, const DATATYPE *input, int batch,
                           int M, int N) {
  uint3 grid = {(M + blockM - 1) / blockM, (N + blockN * accessSize - 1) / (blockN * accessSize), batch};
  uint3 block = {blockM, blockN, 1};
  assert(N % 8 == 0);
  my_row_col_kernel1<<<grid, block>>>(output, input, batch, M, N);
}
