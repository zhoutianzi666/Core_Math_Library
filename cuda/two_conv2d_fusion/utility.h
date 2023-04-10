#include <cuda_fp16.h>

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
} ActType;

typedef struct {
  const half *input;
  const half *weight;
  const half *bias;
  const half *residual;
  const half *b2b_bias;
  const half *b2b_scale;

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
  cudaStream_t stream;
  float alpha;  // for leaky_relu use
  ActType act_type;
} ConvAllParams;



int nchw(struct logical_struct shape, struct logical_struct index);
int nhwc(struct logical_struct shape, struct logical_struct index);


template <typename T>
void naive_conv_cpu(ConvAllParams params, T data);

void cutlass_nhwc_conv_bias(ConvAllParams params, ConvAllParams params1);
