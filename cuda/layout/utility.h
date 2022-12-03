#include <cuda_fp16.h>
void init(float *a, int size);
void init(half *a, int size);
float diff(const half *c, const half *c_baseline, int n);
void naive_nchw_nhwc_cpu(const half *input, half *output, int batch, int ic, int ih, int iw);
void naive_nhwc_nchw_cpu(const half *input, half *output, int batch, int ic, int ih, int iw);
struct logical_struct {
  int n;
  int c;
  int h;
  int w;
};



int nchw(struct logical_struct shape, struct logical_struct index);
int nhwc(struct logical_struct shape, struct logical_struct index);
void my_naive_nchw_nhwc(const half *input, half *output, int batch, int ic, int ih, int iw);
void my_naive_nchw_nhwc1(half *input, half *output, int batch, int ic, int hw);
void cutlass_nchw_nhwc(const half *input, half *output, int batch, int ic, int ih, int iw);
void cutlass_nhwc_nchw(const half *input, half *output, int batch, int ic, int ih, int iw);

void my_row_col(const half *input, half *output, int batch, int m, int n);
