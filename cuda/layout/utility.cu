#include <stdio.h>

#include "utility.h"

void init(half *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
  }
}

float diff(const half *c, const half *c_baseline, int n) {
  float max_diff = -1.;
  for (int i = 0; i < n; i++) {
    float c_value = __half2float(c[i]);
    float base = __half2float(c_baseline[i]);
    if (std::abs(base - c_value) > max_diff) {
      max_diff = std::abs(base - c_value);
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

void naive_nchw_nhwc_cpu(const half *input, half *output, int batch, int ic,
                         int ih, int iw) {
  struct logical_struct input_shape {
    batch, ic, ih, iw
  };
  struct logical_struct output_shape {
    batch, ic, ih, iw
  };
  for (int batch_i = 0; batch_i < batch; batch_i++) {
    for (int ic_i = 0; ic_i < ic; ic_i++) {
      for (int ih_i = 0; ih_i < ih; ih_i++) {
        for (int iw_i = 0; iw_i < iw; iw_i++) {
          struct logical_struct input_index {
            batch_i, ic_i, ih_i, iw_i
          };
          struct logical_struct output_index {
            batch_i, ic_i, ih_i, iw_i
          };
          *(output + nhwc(output_shape, output_index)) =
              *(input + nchw(input_shape, input_index));
        }
      }
    }
  }
}
