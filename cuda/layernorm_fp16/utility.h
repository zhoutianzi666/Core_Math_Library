#include <cuda_fp16.h>
#include "cublas_v2.h"

void init(half *a, int size);

float diff(const half *c, const float *c_baseline, int n);


void  naive_layernorm_cpu(float *output, const half *input, int nc, int h, float epsilon);

template <typename T>
int32_t layernorm_gpu(int32_t const gridSize,
                         int32_t const nHiddenDimension,
                         T const* input,
                         T const* gamma,
                         T const* beta,
                         T* output,
                         float const epsilon);