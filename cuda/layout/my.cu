#include "utility.h"
#include <stdio.h>
using DATATYPE = half;
#define blockM 32
#define blockN 16
#define accessSize 8

__global__ void my_row_col_kernel(const DATATYPE *input,
                                             DATATYPE *output, int batch, int M,
                                             int N) {
  
  __shared__ float4 aTile[blockM][blockN];
  __shared__ float4 bTile[blockN * accessSize ][blockM / accessSize];
  int vol = M * N;
  int batch_i = blockIdx.z;
  // row_0 , col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是16和16*8的倍数！！
  int row_0 = blockIdx.y * blockDim.y;
  int col_0 = blockIdx.x * blockDim.x * 8;
  

  int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  int id_col = thread_id_in_block % blockN ;
  int id_row = thread_id_in_block / blockN ;

  int row_i = row_0 + id_row;
  int col_i = col_0 + id_col * 8;

  int input_offset = batch_i * vol + row_i * N + col_i; 
  aTile[id_row][id_col] = *(float4*)(input + input_offset);

  __syncthreads();
  
  // 将aTile_tmp想象成blockM * blockNN * accessSize
  int a_row = blockM;
  int a_col = blockN * accessSize;
  int b_row = blockN * accessSize;
  int b_col = blockM;
  half* aTile_tmp = (half*)(aTile);
  half* bTile_tmp = (half*)(bTile);
  bTile_tmp += (id_col * accessSize + 0) * b_col + id_row;
  aTile_tmp += id_row * a_col + id_col * accessSize + 0;

  for (int i = 0; i < accessSize; i++) {
    bTile_tmp[i * b_col] = aTile_tmp[i];
  }
  __syncthreads();

  //   // 要变成col_i row_i 了
  int new_idx = thread_id_in_block % (blockM / accessSize);
  int new_idy = thread_id_in_block / (blockM / accessSize);
  row_i = col_0 + new_idy;
  col_i = row_0 + new_idx * 8;

  int output_offset = output_offset = batch_i * vol + row_i * M + col_i;
  *(float4 *)(output + output_offset ) = bTile[new_idy][new_idx];
}

/*
思路是：将输入分成三个部分，hw，c， n这三个部分
对于每个batch，其实就是将行矩阵存储变成列存储矩阵！
hw部分每次派发M个线程
c部分每次派发N个线程
这样每个block需要将M*N个数据从
*/

void my_row_col(const DATATYPE *input, DATATYPE *output, int batch,
                           int M, int N) {
  uint3 grid = {(blockN + N - 1) / blockN / accessSize, (blockM + M - 1) / blockM, batch};
  uint3 block = {blockN, blockM, 1};
  my_row_col_kernel<<<grid, block>>>(input, output, batch, M, N);
}
