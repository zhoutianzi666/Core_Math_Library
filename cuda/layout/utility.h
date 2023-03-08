#include <cuda_fp16.h>

using DATATYPE = half;

void init(float *a, int size);
void init(half *a, int size);
void init(int *a, int size);
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

void my_naive_nchw_nhwc(half *input, half *output, int batch, int ic, int hw);
void cutlass_nchw_nhwc(const half *input, half *output, int batch, int ic, int ih, int iw);
void cutlass_nhwc_nchw(const half *input, half *output, int batch, int ic, int ih, int iw);

void my_row_col0(half *output, const half *input, int batch, int m, int n);
void my_row_col1(DATATYPE *output, const DATATYPE *input, int batch, int m, int n);
void cutlass_iter(half *output, const half *input, int batch, int m, int n);

void my_copy(DATATYPE *output, const  DATATYPE *input, int batch, int m, int n);

