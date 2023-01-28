#include <stdio.h>

#include "utility.h"

void init(int8_t *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = rand() % 256 - 128;
  }
}

// a : row
// b : col
// c: row
void naive_gemm_cpu(const int8_t *a, const int8_t *b, int32_t *c_cpu_int32, int m,
                    int n, int k) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      int32_t sum = 0;
      for (int ii = 0; ii < k; ii++) {
        sum += a[i * k + ii] * b[ii + j * k];
      }
      c_cpu_int32[i * n + j] = sum;
    }
  }
}

int32_t diff(const int32_t *c, const int32_t *c_baseline, int n) {
  int32_t max_diff = -1;
  for (int i = 0; i < n; i++) {
    if (std::abs(c_baseline[i] - c[i]) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c[i]);
    }
  }
  return max_diff;
}

int32_t diff(const int8_t *c, const int32_t *c_baseline, int n) {
  int32_t max_diff = -1;
  for (int i = 0; i < n; i++) {
    if (std::abs(c_baseline[i] - c[i]) > max_diff) {
      max_diff = std::abs(c_baseline[i] - c[i]);
    }
  }
  return max_diff;
}
