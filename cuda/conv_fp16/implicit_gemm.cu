#pragma once
#include <stdio.h>
#include <mma.h>
using namespace nvcuda;
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

using DATATYPE = half;
using DATATYPE = half;
using ACCU_DATATYPE = float;
#define warp_M 16
#define warp_N 16
#define warp_K 16


__device__ int gpu_nhwc_alias(struct logical_struct shape,
                        struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c + index.w * shape.c + index.c;
}

__global__ void my_implicit_gemm_kernel(const half *input, const half *weight,
                                        const half *bias, half *output, 
                                        int batch, int ic, int ih, int iw, 
                                        int kh, int kw, int oc, 
                                        int pad_h, int pad_w, 
                                        int stride_h, int stride_w, 
                                        int oh, int ow) {
  int M = batch * oh * ow;
  int N = oc;
  int K = ic * kh * kw;
  // 当前thread block需要得到的输出矩阵的行地址和列地址
  int m_0 = blockIdx.x * warp_M;
  int n_0 = blockIdx.y * warp_N;

  if (m_0 >= M || n_0 >= N) return;

  __shared__ half aTile[warp_M][warp_K];
  __shared__ half bTile[warp_K][warp_N];
  __shared__ float cTile[warp_M][warp_N];

for(int i = 0;i<warp_M;i++) {
    for(int j = 0;j<warp_N;j++)
    {
        cTile[i][j]=0;
    }
}

  wmma::fragment<wmma::matrix_a, warp_M, warp_N, warp_K, DATATYPE, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, warp_M, warp_N, warp_K, DATATYPE, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, warp_M, warp_N, warp_K, ACCU_DATATYPE> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);


  struct logical_struct input_shape =  {batch, ic, ih, iw};
  struct logical_struct weight_shape =  {oc, ic, kh, kw};

  // 当前的thread block的第一个数字的四维坐标。
  int batch_0 = m_0 / (oh * ow);
  int oh_0 = (m_0 % (oh * ow) ) / ow;
  int ow_0 = (m_0 % (oh * ow) ) % ow;
  int oc_0 = n_0;
  
  const int tidx = threadIdx.x;
  // 这么多线程去搬数据吧！

  for (int k_0 = 0; k_0 < K; k_0 += warp_K) {
    
    // 每个thread 搬运四个数字
    for(int i = tidx; i < warp_M * warp_K; i += 32) {
        int aTile_row = i / warp_K;
        int aTile_col = i % warp_K;
        aTile[aTile_row][aTile_col] = 0;
        
        int m_real = m_0 + aTile_row;
        int k_real = k_0 + aTile_col;

        int batch_real = m_real / (oh * ow);
        int oh_real = (m_real % (oh * ow) ) / ow;
        int ow_real = (m_real % (oh * ow) ) % ow;
        
        // 卧槽这种映射方式不行呢，why？因为你必须要和权重的映射方式保持一致哦！
        // int ic_real = k_real / (kh * kw);
        // int kh_real = (k_real % (kh * kw)) / kw;
        // int kw_real = (k_real % (kh * kw)) % kw;

        int kh_real = k_real / (kw * ic);
        int kw_real = (k_real % (kw * ic)) / ic;
        int ic_real = (k_real % (kw * ic)) % ic;

        int ih_real = oh_real * stride_h - pad_h + kh_real;
        int iw_real = ow_real * stride_w - pad_w + kw_real;

        if (ih_real < 0 || ih_real >= ih) continue;
        if (iw_real < 0 || iw_real >= iw) continue;

        struct logical_struct input_index { batch_real, ic_real, ih_real, iw_real};
        const half *in_ptr = input + gpu_nhwc_alias(input_shape, input_index);
        //const half *in_ptr = input + m_real * K + k_real;

        aTile[aTile_row][aTile_col] = *in_ptr;
    }

    __syncthreads();

    for(int i = tidx; i < warp_K * warp_N; i += 32) {
        int bTile_row = i / warp_N;
        int bTile_col = i % warp_N;
        int k_real = k_0 + bTile_row;
        int n_real = n_0 + bTile_col;
        bTile[bTile_row][bTile_col] = *(weight + k_real + n_real * K);

        // int ic_real = k_real / (kh * kw);
        // int kh_real = (k_real % (kh * kw)) / kw;
        // int kw_real = (k_real % (kh * kw)) % kw;
        // struct logical_struct weight_index =  {n_real, ic_real, kh_real, kw_real};
        // bTile[bTile_row][bTile_col] = *(weight + gpu_nhwc_alias(weight_shape, weight_index));
    }

    __syncthreads();

    // if(tidx == 0) {
    //     for(int i = 0; i < warp_M ; i++) {
    //         for(int j = 0; j < warp_N ; j++) {
    //             float sum = 0;
    //             for(int k = 0; k < warp_K ; k++) {
    //                 sum += __half2float(aTile[i][k]) * __half2float(bTile[k][j]);
    //             }
    //             cTile[i][j] += (sum);
    //         }
    //     }
    // }

    wmma::load_matrix_sync(a_frag, &aTile[0][0], warp_K);
    wmma::load_matrix_sync(b_frag, &bTile[0][0], warp_N);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }
    wmma::store_matrix_sync(&cTile[0][0], c_frag, warp_N, wmma::mem_row_major);

  for(int i = tidx; i < warp_M * warp_N; i += 32) {
    int cTile_row = i / warp_N;
    int cTile_col = i % warp_N;

    int m_real = m_0 + cTile_row;
    int n_real = n_0 + cTile_col;
    *(output + m_real * N + n_real) = cTile[cTile_row][cTile_col];
  }

}

void my_implicit_gemm_gpu(ConvAllParams params) {

  int batch = params.batch;
  int ih = params.ih;
  int iw = params.iw;
  int ic = params.ic;
  int oc = params.oc;
  int kh = params.kh;
  int kw = params.kw;
  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

  int oh = params.oh;
  int ow = params.ow;
  auto input = params.input;
  auto weight = params.weight;
  auto bias = params.bias;
  auto output = params.output;

  int groups = 1;

  int M = batch * oh * ow;
  int N = oc;
  // 假设一个thread block里面只有一个warp。
  constexpr int blockM = warp_M;
  constexpr int blockN = warp_N;
  uint3 block = {32, 1, 1};

  uint3 grid = { (M + blockM - 1) / blockM, (N + blockN - 1) / blockN, 1};
  // 当前假设threadblock中的每个thread计算一个输出哦！
  
  my_implicit_gemm_kernel<<<grid, block>>>(input, weight, bias, output,
  batch, ic, ih, iw, 
  kh, kw, oc, 
  pad_h0, pad_w0, stride_h, stride_w, 
  oh, ow);
}
