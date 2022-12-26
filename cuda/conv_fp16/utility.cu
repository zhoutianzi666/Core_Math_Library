#include <stdio.h>

#include "utility.h"

void init(half *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
  }
}
void init(float *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (rand() % 9999) / 10000.0 - 0.5;
  }
}

float diff(const half *c, const float *c_baseline, int n) {
  float max_diff = -1.;
  for (int i = 0; i < n; i++) {
    float c_value = __half2float(c[i]);
    if (std::abs(c_baseline[i] - c_value) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c_value);
    }
  }
  return max_diff;
}

float diff(const float *c, const float *c_baseline, int n) {
  float max_diff = -1.;
  for (int i = 0; i < n; i++) {
    float c_value = c[i];
    if (std::abs(c_baseline[i] - c_value) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c_value);
    }
  }
  return max_diff;
}

int nchw(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.c * shape.h * shape.w + index.c * shape.h * shape.w +
         index.h * shape.w + index.w;
}

int nhwc(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c +
         index.w * shape.c + index.c;
}

void naive_conv_cpu(ConvAllParams params) {
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int oc = params.oc;
  int kh = params.kh;
  int kw = params.kw;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;
  int oh = params.oh;
  int ow = params.ow;

  auto input = params.input;
  auto weight = params.weight;
  auto residual = params.residual;
  auto bias = params.bias;
  auto output = params.output_cpu_fp32;

  struct logical_struct input_shape {
    batch, ic, ih, iw
  };
  struct logical_struct output_shape {
    batch, oc, oh, ow
  };
  struct logical_struct weight_shape {
    oc, ic, kh, kw
  };
  for (int bs_i = 0; bs_i < batch; bs_i++) {
    for (int oc_i = 0; oc_i < oc; oc_i++) {
      for (int oh_i = 0; oh_i < oh; oh_i++) {
        for (int ow_i = 0; ow_i < ow; ow_i++) {
          struct logical_struct output_index {
            bs_i, oc_i, oh_i, ow_i
          };
          float *out_ptr = output + nhwc(output_shape, output_index);
          float sum = 0.f;

          for (int kh_i = 0; kh_i < kh; kh_i++) {
            for (int kw_i = 0; kw_i < kw; kw_i++) {
              int ih_i = oh_i * stride_h - pad_h0 + kh_i * dilation_h;
              int iw_i = ow_i * stride_w - pad_w0 + kw_i * dilation_w;
              if (ih_i < 0 || ih_i >= ih) continue;
              if (iw_i < 0 || iw_i >= iw) continue;

              for (int ic_i = 0; ic_i < ic; ic_i++) {
                struct logical_struct input_index {
                  bs_i, ic_i, ih_i, iw_i
                };
                struct logical_struct weight_index {
                  oc_i, ic_i, kh_i, kw_i
                };
                const half *in_ptr = input + nhwc(input_shape, input_index);
                const half *weight_ptr =
                    weight + nhwc(weight_shape, weight_index);
                sum += __half2float(*in_ptr) * __half2float(*weight_ptr);
              }
            }
          }
          if (residual)
            sum += __half2float(*(residual + nhwc(output_shape, output_index)));

          // bias
          sum += __half2float(*(bias + oc_i));

          // no bias
          // *out_ptr = sum;

          // relu
          //*out_ptr = sum > 0 ? sum : 0.f;

          // silu
          float x = sum;
          *out_ptr = (x) * (1.f / (1 + exp(-x)));

          // leaky_relu
          // if (x > 0) *out_ptr = x;
          // else {
          //   *out_ptr = x * 0.5 ;
          // }
        }
      }
    }
  }
}
