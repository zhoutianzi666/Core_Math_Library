#include "utility.h"
using DATATYPE = half;

__global__ void my_naive_kernel(const DATATYPE *input, DATATYPE *output, int batch,
                          int ic, int ih, int iw) {
  // struct logical_struct output_shape{batch, ic, ih, iw};
  int input_offset = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_i = input_offset / (ic * ih * iw);
  int ic_i = (input_offset % (ic * ih * iw)) / (ih * iw);
  int ih_i = ((input_offset % (ic * ih * iw)) % (ih * iw)) / iw;
  int iw_i = ((input_offset % (ic * ih * iw)) % (ih * iw)) % iw;
  // struct logical_struct output_index {batch_i, ic_i, ih_i, iw_i};
  //*(output + nhwc(output_shape, output_index)) = *(input + input_offset);
  int output_offset =
      batch_i * ic * ih * iw + ih_i * iw * ic + iw_i * ic + ic_i;
  *(output + output_offset) = *(input + input_offset);
}

void my_naive_nchw_nhwc(const DATATYPE *input, DATATYPE *output, int batch, int ic,
                  int ih, int iw) {
  int nchw = batch * ic * ih * iw;
  int thread_num = 128;
  uint3 grid = {nchw / thread_num + 1, 1, 1};
  uint3 block = {thread_num, 1, 1};
  my_naive_kernel<<<grid, block>>>(input, output, batch, ic, ih, iw);
}


#include <stdio.h>
using DATATYPE = half;

__global__ void my_naive_kernel1(const DATATYPE *input, DATATYPE *output,
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

void my_naive_nchw_nhwc1(DATATYPE *input, DATATYPE *output, int batch, int m,
                  int n) {
  int blockM = 18;
  int blockN = 32;
  uint3 grid = { (m + blockM - 1) / blockM, (n + blockN - 1) / blockN, batch};
  uint3 block = {blockM, blockN, 1};
  my_naive_kernel1<<<grid, block>>>(input, output, m, n);
}

