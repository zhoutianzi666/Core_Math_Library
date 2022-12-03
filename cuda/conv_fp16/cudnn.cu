#pragma once
#include <stdio.h>

#include <iostream>

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

void cudnn_nhwc_conv(const half *input, const half *weight, half *output,
                     int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                     int pad_h, int pad_w, int stride_h, int stride_w, int oh,
                     int ow) {}
