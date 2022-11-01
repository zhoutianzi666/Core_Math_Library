#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = half;
#include <algorithm>
#define warp_M 16
#define warp_N 8
#define warp_K 8
#define WARP_SIZE 32
using DATATYPE = half;

__global__ void my_kernel(const half *input, const half *weight, half *output,
    int batch, int ic, int ih, int iw, int kh, int kw,
    int oc, int pad_h, int pad_w, int stride_h, int stride_w,
    int oh, int ow) 
{
    int oc_begin = blockIdx.y * warp_N;
    int onhw_begin = blockIdx.x * warp_M;
    __shared__ DATATYPE inputTile[warp_M][warp_K];
    __shared__ DATATYPE weightTile[warp_K][warp_N];

    for (int kh_i = 0; kh_i < kh; kh_i ++) {
        for (int kw_i = 0; kw_i < kw; kw_i ++) {
            for (int ic_i = 0; ic_i < ic; ic_i += warp_K) {
                
            }
        }
    }

}
void my_nhwc_conv(const half *input, const half *weight, half *output,
                       int batch, int ic, int ih, int iw, int kh, int kw,
                       int oc, int pad_h, int pad_w, int stride_h, int stride_w,
                       int oh, int ow) {
// mma.16,8,8
int onhw = batch * oh * ow;
uint3 grid = {onhw / warp_M, oc / warp_N, 1};
// a block having only one warp!
uint3 block = {WARP_SIZE, 1, 1};
my_kernel<<<grid, block>>>(input, weight, output,
                       batch, ic, ih, iw, kh, kw,
                       oc, pad_h, pad_w, stride_h, stride_w,
                       oh, ow);
}
