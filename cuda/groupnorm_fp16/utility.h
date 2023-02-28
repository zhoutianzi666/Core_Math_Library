#include <cuda_fp16.h>
#include "cublas_v2.h"

void init(half *a, int size);

float diff(const half *c, const float *c_baseline, int n);

void naive_groupnorm_cpu(float *output, const half *input, int n, int c, int h,
                         int w, int groups);
void groupnorm_gpu(half *output, const half *input, int n, int c, int h,
    int w, int groups);
