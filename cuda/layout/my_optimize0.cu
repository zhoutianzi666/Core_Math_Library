#include "utility.h"
#include <stdio.h>
using DATATYPE = half;
#define blockM 16
#define blockN 16
#define accessSize 8

__global__ void my_row_col_kernel0(const DATATYPE *input,
                                             DATATYPE *output, int batch, int M,
                                             int N) {
  
  __shared__ float4 aTile[blockM][blockN];
  __shared__ float4 bTile[blockN * accessSize ][blockM / accessSize];
  int vol = M * N;
  int batch_i = blockIdx.z;

  // g_row_0 , g_col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是16和16*8的倍数！！
  int g_row_0 = blockIdx.x * blockDim.x;
  int g_col_0 = blockIdx.y * blockDim.y * accessSize;

  int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  // local column and row read by the current cuda thread
  int local_rowi = thread_id_in_block % 16;
  int local_coli = thread_id_in_block / 16;

  int g_row_i = g_row_0 + local_rowi;
  int g_col_i = g_col_0 + local_coli * accessSize;

  int input_offset = batch_i * vol + g_row_i * N + g_col_i; 
  aTile[local_rowi][local_coli] = *(float4*)(input + input_offset);

  __syncthreads();
  
  // 将aTile_tmp想象成blockM * blockNN * accessSize
  int a_row = blockM;
  int a_col = blockN * accessSize;
  int b_row = blockN * accessSize;
  int b_col = blockM;
  half* aTile_tmp = (half*)(aTile);
  half* bTile_tmp = (half*)(bTile);

  for (int i = 0; i < accessSize; i++) {
    bTile_tmp[(local_coli * accessSize + i) * b_col + local_rowi] = aTile_tmp[local_rowi * a_col + local_coli * accessSize + i];
  }
  __syncthreads();

  //   // 要变成col_i row_i 了
  int new_idx = thread_id_in_block % 2;
  int new_idy = thread_id_in_block / 2;
  g_row_i = g_col_0 + new_idy;
  g_col_i = g_row_0 + new_idx * accessSize;

  int output_offset = output_offset = batch_i * vol + g_row_i * M + g_col_i;
  *(float4 *)(output + output_offset ) = bTile[new_idy][new_idx];

}

/*

思路是：将输入分成三个部分，M，N， batch这三个部分
每个thread block计算 blockM * (blockN * accessSize) 这么大部分！
每个thread block的线程维度是(blockM, blockN)
对于每个batch，其实就是将行矩阵存储变成列存储矩阵！
*/

void my_row_col0(DATATYPE *output, const DATATYPE *input, int batch,
                           int M, int N) {
  uint3 grid = {(blockM + M - 1) / blockM, (blockN + N - 1) / blockN / accessSize, batch};
  uint3 block = {blockM, blockN, 1};
  my_row_col_kernel0<<<grid, block>>>(input, output, batch, M, N);
}
