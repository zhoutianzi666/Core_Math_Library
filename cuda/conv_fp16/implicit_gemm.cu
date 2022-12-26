// #pragma once
// #include <stdio.h>

// #include <algorithm>
// #include <chrono>
// #include <ctime>
// #include <iostream>
// #include <ratio>

// #include "cublas_v2.h"
// #include "cutlass/gemm/device/gemm.h"
// #include "utility.h"

// using DATATYPE = half;
// using DATATYPE = half;

// __device__ int gpu_nhwc_alias(struct logical_struct shape,
//                         struct logical_struct index) {
//   return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c
//   +
//          index.w * shape.c + index.c;
// }

// __global__ void my_implicit_gemm_kernel(const half *input, const half
// *weight,
//                           const half *bias, half *output, int batch, int ic,
//                           int ih, int iw, int kh, int kw, int oc, int pad_h,
//                           int pad_w, int stride_h, int stride_w, int oh,
//                           int ow) {
//   int M = batch * oh * ow;
//   int N = oc;
//   int K = ic * kh * kw;
//   int m_i = threadIdx.x + blockIdx.x * blockDim.x;
//   int n_i = threadIdx.y + blockIdx.y * blockDim.y;

//   if (m_i >= M || n_i >= N) return;

//   struct logical_struct weight_shape = {oc, ic, kh, kw};
//   struct logical_struct input_shape =  { batch, ic, ih, iw};
//   int batch_i = m_i / (oh * ow);
//   int oh_i = (m_i % (oh * ow) ) / ow;
//   int ow_i = (m_i % (oh * ow) ) % ow;
//   int oc_i = n_i;

//   half *out_ptr = output + m_i * N + n_i;
//   float sum = 0.f;

//   for (int k_i = 0; k_i < K; k_i++) {
//     int ic_i = k_i / (kh * kw);
//     int kh_i = (k_i % (kh * kw)) / kw;
//     int kw_i = (k_i % (kh * kw)) % kw;
//     struct logical_struct weight_index { oc_i, ic_i, kh_i, kw_i };
//     const half *weight_ptr = weight + gpu_nhwc_alias(weight_shape,
//     weight_index);

//     int ih_i = oh_i * stride_h - pad_h + kh_i;
//     int iw_i = ow_i * stride_w - pad_w + kw_i;

//     if (ih_i < 0 || ih_i >= ih) continue;
//     if (iw_i < 0 || iw_i >= iw) continue;

//     struct logical_struct input_index { batch_i, ic_i, ih_i, iw_i };
//     const half *in_ptr = input + gpu_nhwc_alias(input_shape, input_index);
//     sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
//   }

//   sum += __half2float(*(bias + oc_i));

//   // silu
//   float x = sum;
//   *out_ptr = __float2half(x * (1.f / (1 + exp(-x))));
// }

// void my_implicit_gemm_gpu(const half *input, const half *weight, const half
// *bias,
//                        half *output, int batch, int ic, int ih, int iw, int
//                        kh, int kw, int oc, int pad_h, int pad_w, int
//                        stride_h, int stride_w, int oh, int ow) {
//   int M = batch * oh * ow;
//   int N = oc;
//   constexpr int blockM = 16;
//   constexpr int blockN = 16;

//   uint3 grid = { (M + blockM - 1) / blockM, (N + blockN - 1) / blockN, 1};
//   // 当前假设threadblock中的每个thread计算一个输出哦！
//   uint3 block = {blockM, blockN, 1};

//   my_implicit_gemm_kernel<<<grid, block>>>(input, weight, bias, output,
//   batch, ic, ih, iw, kh,
//                              kw, oc, pad_h, pad_w, stride_h, stride_w, oh,
//                              ow);
// }
