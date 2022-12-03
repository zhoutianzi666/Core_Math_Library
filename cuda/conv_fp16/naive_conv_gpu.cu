#pragma once
#include <stdio.h>

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

__device__ int gpu_nhwc(struct logical_struct shape,
                        struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c +
         index.w * shape.c + index.c;
}

__global__ void my_kernel(const half *input, const half *weight,
                          const half *bias, half *output, int batch, int ic,
                          int ih, int iw, int kh, int kw, int oc, int pad_h,
                          int pad_w, int stride_h, int stride_w, int oh,
                          int ow) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int batch_i = idx / (oc * oh * ow);
  int remain = idx % (oc * oh * ow);
  int oc_i = remain / (oh * ow);
  remain = idx % (oh * ow);
  int oh_i = remain / ow;
  int ow_i = remain % ow;
  struct logical_struct input_shape {
    batch, ic, ih, iw
  };
  struct logical_struct output_shape = {batch, oc, oh, ow};
  struct logical_struct output_index = {batch_i, oc_i, oh_i, ow_i};
  struct logical_struct weight_shape = {oc, ic, kh, kw};
  half *out_ptr = output + gpu_nhwc(output_shape, output_index);
  float sum = 0.f;

  for (int kh_i = 0; kh_i < kh; kh_i++) {
    for (int kw_i = 0; kw_i < kw; kw_i++) {
      int ih_i = oh_i * stride_h - pad_h + kh_i;
      int iw_i = ow_i * stride_w - pad_w + kw_i;
      if (ih_i < 0 || ih_i >= ih) continue;
      if (iw_i < 0 || iw_i >= iw) continue;

      for (int ic_i = 0; ic_i < ic; ic_i++) {
        struct logical_struct input_index {
          batch_i, ic_i, ih_i, iw_i
        };
        struct logical_struct weight_index {
          oc_i, ic_i, kh_i, kw_i
        };
        const half *in_ptr = input + gpu_nhwc(input_shape, input_index);
        const half *weight_ptr = weight + gpu_nhwc(weight_shape, weight_index);
        sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
      }
    }
  }
  sum += __half2float(*(bias + oc_i));

  // silu
  float x = sum;
  *out_ptr = __float2half(x * (1.f / (1 + exp(-x))));
}

void my_naive_conv_gpu(const half *input, const half *weight, const half *bias,
                       half *output, int batch, int ic, int ih, int iw, int kh,
                       int kw, int oc, int pad_h, int pad_w, int stride_h,
                       int stride_w, int oh, int ow) {
  // mma.16,8,8
  int onhwc = batch * oh * ow * oc;
  uint3 grid = {onhwc / 256, 1, 1};
  // 每个block计算256个数字
  uint3 block = {256, 1, 1};

  my_kernel<<<grid, block>>>(input, weight, bias, output, batch, ic, ih, iw, kh,
                             kw, oc, pad_h, pad_w, stride_h, stride_w, oh, ow);
}
