#pragma once
#include <stdio.h>
#include <iostream>
#include "utility.h"


__global__ void kernel1(ConvAllParams params)
{
    int batch_i = blockIdx.x;
    int ic_i = blockIdx.y;
    int ic = params.ic;
    int ih = params.ih;
    int iw = params.iw;
    int oc_i = blockIdx.y;
    int oc = params.oc;
    int oh = params.oh;
    int ow = params.ow;
    int ohw_start = threadIdx.x;
    int ohw = oh * ow;
    int kh = 3;//params.kh;
    int kw = 3;//params.kw;
    auto input = params.input;
    auto weight = params.weight;
    auto bias = params.bias;
    auto output = params.output;

  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

    for(int ohw_i = ohw_start; ohw_i < ohw; ohw_i+= blockDim.x)
    {
        int oh_i = ohw_i / ow;
        int ow_i = ohw_i % ow;
        half* now_out_ptr = output + batch_i * oc * oh * ow + oc_i * oh * ow + ohw_i;
        float sum = 0.f;

        for (int kh_i = 0; kh_i < kh; kh_i ++)
        {

            for (int kw_i = 0; kw_i < kw; kw_i ++)
            {
                int ih_i = oh_i * stride_h - pad_h0 + kh_i * dilation_h;
                int iw_i = ow_i * stride_w - pad_w0 + kw_i * dilation_w;
                if (ih_i < 0 || ih_i >= ih) continue;
                if (iw_i < 0 || iw_i >= iw) continue;

                const half* now_input_ptr = input + batch_i * ic * ih * iw + ic_i * ih * iw + ih_i * iw + iw_i;
                const half* now_k_ptr = weight + oc_i * kh * kw + kh_i * kw + kw_i;
                sum += (float)(*now_k_ptr) * (float)(*now_input_ptr);
            }
        }
        sum += (float)(*(bias + oc_i));
        //sum = 1/ (1 + std::exp(-sum)) ;
        *now_out_ptr = (half)sum;
    }
}

__global__ void kernel2(ConvAllParams params)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int batch = params.batch;
    int ic = params.ic;
    int ih = params.ih;
    int iw = params.iw;
    int oc = params.oc;
    int oh = params.oh;
    int ow = params.ow;
    if (idx >= batch * oc * oh * ow) return;

    int batch_i = idx / (oc * oh * ow);
    int tmp = idx % (oc * oh * ow);
    int oc_i = tmp / (oh * ow);
    tmp = tmp % (oh * ow);
    int oh_i = tmp / ow;
    int ow_i = tmp % ow;

    int ic_i = oc_i;
    int kh = 3;
    int kw = 3;
    auto input = params.input;
    auto weight = params.weight;
    auto bias = params.bias;
    auto output = params.output;

  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

    for(int ii = 0; ii < 1; ii++)
    {
        half* now_out_ptr = output + idx;
        float sum = 0.f;

        for (int kh_i = 0; kh_i < kh; kh_i ++)
        {
            for (int kw_i = 0; kw_i < kw; kw_i ++)
            {
                int ih_i = oh_i * stride_h - pad_h0 + kh_i * dilation_h;
                int iw_i = ow_i * stride_w - pad_w0 + kw_i * dilation_w;
                if (ih_i < 0 || ih_i >= ih) continue;
                if (iw_i < 0 || iw_i >= iw) continue;

                const half* now_input_ptr = input + batch_i * ic * ih * iw + ic_i * ih * iw + ih_i * iw + iw_i;
                const half* now_k_ptr = weight + oc_i * kh * kw + kh_i * kw + kw_i;
                sum += (float)(*now_k_ptr) * (float)(*now_input_ptr);
            }
        }
        sum += (float)(*(bias + oc_i));
        *now_out_ptr = (half)sum;
    }
}


#include <algorithm>
void  my_nchw_cov2d_depthwise(ConvAllParams params)
{
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
  int groups = params.groups;
// 方法1:2个block计算一个output channel
  uint3 grid = { batch , oc, 1};
  uint3 block = {512, 1, 1};
  kernel1<<<grid, block>>>(params);
  // 方法2:每个 cuda thread计算一个输出数字
  // int thread_in_block = 256;
  // uint3 block = {thread_in_block, 1, 1};
  // uint3 grid = {(batch * oc * oh * ow) / thread_in_block + 1 , 1, 1};
  // kernel2<<<grid, block>>>(params);
}
