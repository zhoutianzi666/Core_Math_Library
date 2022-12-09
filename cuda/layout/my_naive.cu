#include "utility.h"
#include <stdio.h>

using DATATYPE = half;

__global__ void my_naive_kernel(const DATATYPE *input, DATATYPE *output,
                          int m, int n) {
  int vol = m * n;
  int batch_i = blockIdx.z;
  int row_i = threadIdx.x + blockIdx.x * blockDim.x;
  int col_i = threadIdx.y + blockIdx.y * blockDim.y;
  if(row_i >= m || col_i >=n) return;
  int input_offset = batch_i * vol + row_i * n + col_i;
  int output_offset = batch_i * vol + row_i + col_i * m;
  *(output + output_offset) = *(input + input_offset);
}

void my_naive_nchw_nhwc(DATATYPE *input, DATATYPE *output, int batch, int m,
                  int n) {
  const int blockM = 18;
  const int blockN = 32;
  uint3 grid = { (m + blockM - 1) / blockM, (n + blockN - 1) / blockN, batch};
  uint3 block = {blockM, blockN, 1};
  my_naive_kernel<<<grid, block>>>(input, output, m, n);
}


