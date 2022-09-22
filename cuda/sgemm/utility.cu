#include <stdio.h>

#include "utility.h"

void init(float *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = (rand() % 9999) / 10000.0;
  }
}

void naive_gemm_cpu(const float *a, const float *b, float *c_cpu_fp32, int m,
                    int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.f;
      for (int ii = 0; ii < k; ii++) {
        sum += a[i * k + ii] * b[ii * n + j];
      }
      c_cpu_fp32[i * n + j] = sum;
    }
  }
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