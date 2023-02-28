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
  int m;
  int n;
};

int rowmajor(struct logical_struct shape, struct logical_struct index) {
  return index.m * shape.n + index.n;
}

void  naive_layernorm_cpu(float *output, const half *input, int m, int n, float epsilon)
{
  struct logical_struct shape {m, n};

  for (int i = 0; i < m ;i++)
  {
    float sum = 0.f;
    float q2sum = 0.f;
    for (int j = 0; j < n; j++)
    {
      struct logical_struct index {i, j};
      float tmp_data = *(input + rowmajor(shape, index));
      sum += tmp_data;
      q2sum += tmp_data * tmp_data;;
    }

    float mean = sum / n;
    float sigma = sqrtf(q2sum / n - mean * mean + epsilon);

    for (int j = 0; j < n; j++)
    {
      struct logical_struct index {i, j};
      float tmp_data = *(input + rowmajor(shape, index));
      float norm_data = (tmp_data - mean) / sigma;
      // 这里为了减少参数个数，选择将输入作为那个！
      float scale = *(input + j);
      float bias = *(input + j);

      norm_data = norm_data * scale + bias;
      *(output + rowmajor(shape, index)) = norm_data; 
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