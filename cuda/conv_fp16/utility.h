#include <cuda_fp16.h>
void init(float *a, int size);
void init(half *a, int size);
float diff(const half *c, const float *c_baseline, int n);
float diff(const float *c, const float *c_baseline, int n);

void naive_conv_cpu(const half *input, const half *weight, float *output,
                    int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                    int pad_h, int pad_w, int stride_h, int stride_w);

void cutlass_nhwc_conv(const half *input, const half *weight, half *output,
                  int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                  int pad_h, int pad_w, int stride_h, int stride_w, int oh,
                  int ow);
