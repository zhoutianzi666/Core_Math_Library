#include "utility.h"
#include <stdio.h>
#include "assert.h" 

using DATATYPE = half;

using DATATYPE2 = half2;

#define blockM 32
#define blockN 8
#define padding 2
#define accessSize 8

__global__ void my_copy_kernel0(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  int vol = M * N;
  int batch_i = blockIdx.z;
  
  // g_row_0 , g_col_0表示这个input矩阵的block的最左上角的全局的行号和列号，他们分别是blockM和blockN*accessSize的倍数
  // 一定要记住了，这里选择用blockIdx.y乘是有深意的！
  int g_row_0 = blockIdx.y * blockDim.x;
  int g_col_0 = blockIdx.x * blockDim.y * accessSize;

  int thread_id_in_block = threadIdx.x + threadIdx.y * blockDim.x;
  // local column and row read by the current cuda thread
  int local_rowi = thread_id_in_block / blockN;
  int local_coli = thread_id_in_block % blockN;

  int g_row_i = g_row_0 + local_rowi;
  int g_col_i = g_col_0 + local_coli * accessSize;
  int input_offset = batch_i * vol + g_row_i * N + g_col_i; 
  *(float4*)(output + input_offset ) = *(float4*)(input + input_offset);

}

__global__ void my_copy_kernel1(DATATYPE *output, 
                                   const DATATYPE *input,
                                             int batch, int M,
                                             int N) {
  int offset = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
  *(float4*)(output + offset ) = *(float4*)(input + offset);
}

void my_copy(DATATYPE *output, const DATATYPE *input, int batch,
                           int M, int N) {
  assert(N % 8 == 0);
  assert(M % 8 == 0);

  // 这个是把输入想象成一个矩阵来拷贝
  // uint3 grid = {(N + blockN * accessSize - 1) / (blockN * accessSize), (M + blockM - 1) / blockM, batch};
  // uint3 block = {blockM, blockN, 1};
  // my_copy_kernel0<<<grid, block>>>(output, input, batch, M, N);

  // 这个是把输入想象成一个一维数组来拷贝
  uint3 grid = {(M * N) / (32 * 8), 1, 1};
  uint3 block = {32, 1, 1};
  my_copy_kernel1<<<grid, block>>>(output, input, batch, M, N);

}






