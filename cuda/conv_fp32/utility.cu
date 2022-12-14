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

struct logical_struct {
  int n;
  int c;
  int h;
  int w;
};

int nchw(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.c * shape.h * shape.w + index.c * shape.h * shape.w +
         index.h * shape.w + index.w;
}

int nhwc(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.h * shape.w * shape.c + index.h * shape.w * shape.c +
         index.w * shape.c + index.c;
}

void naive_conv_cpu(const float *input, const float *weight, float *output,
                    int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                    int pad_h, int pad_w, int stride_h, int stride_w) {
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
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
              int ih_i = oh_i * stride_h - pad_h + kh_i;
              int iw_i = ow_i * stride_w - pad_w + kw_i;
              if (ih_i < 0 || ih_i >= ih) continue;
              if (iw_i < 0 || iw_i >= iw) continue;

              for (int ic_i = 0; ic_i < ic; ic_i++) {
                struct logical_struct input_index {
                  bs_i, ic_i, ih_i, iw_i
                };
                struct logical_struct weight_index {
                  oc_i, ic_i, kh_i, kw_i
                };
                const float *in_ptr = input + nhwc(input_shape, input_index);
                const float *weight_ptr =
                    weight + nhwc(weight_shape, weight_index);
                sum += (*in_ptr) * (*weight_ptr);
              }
            }
          }
          *out_ptr = sum;
        }
      }
    }
  }
}
