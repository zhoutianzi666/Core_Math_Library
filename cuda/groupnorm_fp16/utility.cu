#include <assert.h>
#include <stdio.h>

#include <iostream>

#include "utility.h"

void init(half *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = __float2half((rand() % 9999) / 10000.0 - 0.5);
    a[i] = __float2half((rand() % 4));
  }
}

struct logical_struct {
  int n;
  int c;
  int h;
  int w;
};

//
int nchw32(struct logical_struct shape, struct logical_struct index) {
  return index.n * shape.c / 32 * shape.h * shape.w * 32 +
         index.c / 32 * shape.h * shape.w * 32 + index.h * shape.w * 32 +
         index.w * 32 + index.c % 32;
}

// 假设输入是nchw32
void naive_groupnorm_cpu(float *output, const half *input, int n, int c, int h,
                         int w, int groups) {
  assert(c % 32 == 0);
  assert(c % groups == 0);

  struct logical_struct shape {
    n, c, h, w
  };

  for (int ni = 0; ni < n; ni++) {
    for (int group_i = 0; group_i < groups; group_i++) {
      
      int ci_begin = group_i * (c / groups);
      int ci_end = (group_i + 1) * (c / groups);

      float sum = 0.f;
      float q2sum = 0.f;

      for (int ci = ci_begin; ci < ci_end; ci++) {
        for (int hi = 0; hi < h; hi++) {
          for (int wi = 0; wi < w; wi++) {
            struct logical_struct index {
              ni, ci, hi, wi
            };
            float tmp_data = *(input + nchw32(shape, index));
            sum += tmp_data;
            q2sum += tmp_data * tmp_data;
          }
        }
      }

      int nums = h * w * c / groups;
      float mean = sum / nums;
      float sigma = sqrtf(q2sum / nums - mean * mean);

      for (int ci = ci_begin; ci < ci_end; ci++) {
        for (int hi = 0; hi < h; hi++) {
          for (int wi = 0; wi < w; wi++) {
            struct logical_struct index {
              ni, ci, hi, wi
            };
            float tmp_data = *(input + nchw32(shape, index));
            float norm_data = (tmp_data - mean) / sigma;
            *(output + nchw32(shape, index)) = norm_data; 
          }
        }
      }

    }
  }

  // 下面肯定是要搞beta和gamma了，我懒得写了
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          struct logical_struct index {
            ni, ci, hi, wi
          };
          float tmp_data = *(output + nchw32(shape, index));
        }
      }
    }
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