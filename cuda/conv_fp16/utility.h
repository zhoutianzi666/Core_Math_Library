#include <cuda_fp16.h>
#include <cudnn_v8.h>
#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include "cutlass/cutlass.h"

#define CUTLASS_CHECK(status)                                               \
  if (status != cutlass::Status::kSuccess) {                                \
    std::cout                                                               \
        << "Cutlass can not deal with this problem size, skip this kernel!" \
        << std::endl;                                                       \
  }

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

typedef enum {
  IDENTITY,
  RELU,
  SILU,
  LEAKY_RELU,
  SIGMOID,
  CONV2D_BIAS_ADD_RELU,
} ActType;

typedef struct {
  const half *input;
  const half *weight;
  const half *bias;
  const half *residual;
  half *output;
  float *output_cpu_fp32;
  int batch;
  int ic;
  int ih;
  int iw;
  int kh;
  int kw;
  int oc;
  int pad_h0;
  int pad_h1;
  int pad_w0;
  int pad_w1;
  int stride_h;
  int stride_w;
  int dilation_h;
  int dilation_w;
  int oh;
  int ow;
  int groups = 1;
  cudaStream_t stream;
  float alpha;  // for leaky_relu use
  ActType act_type;

  // for cuDNN use
  cudnnHandle_t handle_cudnn;
  int cudnn_workspace_size;
  void *cudnn_workspace;
} ConvAllParams;

int nchw(struct logical_struct shape, struct logical_struct index);
int nhwc(struct logical_struct shape, struct logical_struct index);

void naive_conv_cpu(ConvAllParams params);

void cutlass_nhwc_conv_relu(ConvAllParams params);

void cutlass_nhwc_conv_residual(ConvAllParams params);
void cutlass_nhwc_conv_bias_swish(ConvAllParams params);


void my_naive_conv_gpu(const half *input, const half *weight, const half *bias,
                       half *output, int batch, int ic, int ih, int iw, int kh,
                       int kw, int oc, int pad_h, int pad_w, int stride_h,
                       int stride_w, int oh, int ow);

void my_implicit_gemm_gpu(const half *input, const half *weight,
                          const half *bias, half *output, int batch, int ic,
                          int ih, int iw, int kh, int kw, int oc, int pad_h,
                          int pad_w, int stride_h, int stride_w, int oh,
                          int ow);
void cutlass_nhwc_conv_bias_leaky_relu(ConvAllParams params);

void cutlass_nhwc_conv_bias_swish_simt(ConvAllParams params);
void cutlass_nhwc_conv_depthwise(ConvAllParams params); 
void cutlass_group_conv_bias_swish(ConvAllParams params);

std::string cudnnAlgoName(cudnnConvolutionFwdAlgo_t algo);
void CUDNN_CHECK(cudnnStatus_t status);
void cudnn_nhwc_conv(ConvAllParams params);







int ProfileToGetBestConfig(
    const std::vector<std::function<cutlass::Status(ConvAllParams)>> &all_func,
    const ConvAllParams &params);

