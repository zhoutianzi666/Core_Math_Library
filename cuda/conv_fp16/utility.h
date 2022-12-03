#include <cuda_fp16.h>

void init(float *a, int size);
void init(half *a, int size);
float diff(const half *c, const float *c_baseline, int n);
float diff(const float *c, const float *c_baseline, int n);


struct logical_struct {
  int n;
  int c;
  int h;
  int w;
};

int nchw(struct logical_struct shape, struct logical_struct index);
int nhwc(struct logical_struct shape, struct logical_struct index);


void naive_conv_cpu(const half *input, const half *weight, const half * bias, float *output,
                    int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                    int pad_h, int pad_w, int stride_h, int stride_w, const half *residual = nullptr);


void cutlass_nhwc_conv(const half *input, const half *weight, const half *bias, half *output,
                  int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                  int pad_h, int pad_w, int stride_h, int stride_w, int oh,
                  int ow);


void cutlass_nhwc_conv_residual(const half *input, const half *weight, const half *bias, half *output,
                  int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                  int pad_h, int pad_w, int stride_h, int stride_w, int oh,
                  int ow, const half *residual);

void cutlass_nhwc_conv_bias_swish(const half *input, const half *weight, const half *bias, half *output,
                  int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                  int pad_h, int pad_w, int stride_h, int stride_w, int oh,
                  int ow);

void my_naive_conv_gpu(const half *input, const half *weight, 
                  const half *bias,
                  half *output,
                  int batch, int ic, int ih, int iw, int kh, int kw, int oc,
                  int pad_h, int pad_w, int stride_h, int stride_w, int oh, int ow);