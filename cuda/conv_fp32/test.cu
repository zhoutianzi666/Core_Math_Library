
#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include <cudnn_v8.h>
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = float;
using C_DATATYPE = float;

int main(void) {
  int batch = 1;
  int ic = 32;
  int ih = 112;
  int iw = 112;
  int pad_h = 1;
  int pad_w = 1;
  int oc = 32;
  int kh = 3;
  int kw = 3;
  int stride_h = 1;
  int stride_w = 1;

  // Here is consistent with
  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

  // Note input and weight is in CPU place
  DATATYPE *input, *weight;
  int input_size = batch * ic * ih * iw;
  int weight_size = oc * ic * kh * kw;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  status = cudaMallocHost(&weight, sizeof(DATATYPE) * weight_size);
  init(input, input_size);
  init(weight, weight_size);

  // out is used to store the result form dev_out
  C_DATATYPE *out_from_dev;
  int out_size = batch * oc * oh * ow;
  cudaMallocHost(&out_from_dev, sizeof(C_DATATYPE) * out_size);
  memset(out_from_dev, 0, sizeof(C_DATATYPE) * out_size);

  // dev_input is from weight
  // dev_weight is from weight
  DATATYPE *dev_input, *dev_weight;
  C_DATATYPE *dev_out;

  cudnnHandle_t handle_cudnn;
  cudnnCreate(&handle_cudnn);
  cudnnTensorDescriptor_t input_descriptor;
  auto cudnn_state = cudnnCreateTensorDescriptor(&input_descriptor);
  cudnnSetTensor4dDescriptor(input_descriptor,
                             CUDNN_TENSOR_NHWC,
                             CUDNN_DATA_FLOAT, batch, ic, ih, iw);
  cudnnTensorDescriptor_t output_descriptor;
  cudnn_state=cudnnCreateTensorDescriptor(&output_descriptor);
  cudnn_state=cudnnSetTensor4dDescriptor(output_descriptor,
                            CUDNN_TENSOR_NHWC,
                            CUDNN_DATA_FLOAT, batch, oc, oh, ow);
  cudnnFilterDescriptor_t kernel_descriptor;
  cudnn_state=cudnnCreateFilterDescriptor(&kernel_descriptor);
  cudnn_state=cudnnSetFilter4dDescriptor(kernel_descriptor,
                              CUDNN_DATA_FLOAT,
                              CUDNN_TENSOR_NHWC,
                              oc, ic, kh, kw);

  cudnnConvolutionDescriptor_t conv_descriptor;
  cudnn_state= cudnnCreateConvolutionDescriptor(&conv_descriptor);
  cudnn_state = cudnnSetConvolution2dDescriptor(conv_descriptor,
                                  1, 1, // zero-padding
                                  1, 1, // stride
                                  1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);


int returnedAlgoCount;
cudnnConvolutionFwdAlgoPerf_t     perfResults[100];
cudnnFindConvolutionForwardAlgorithm(handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
  output_descriptor, 100, &returnedAlgoCount, perfResults);


  printf("%d\n", returnedAlgoCount);
size_t workspace_size = 0;
cudnn_state = cudnnGetConvolutionForwardWorkspaceSize(handle_cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        conv_descriptor,
                                        output_descriptor,
                                        perfResults[0].algo,
                                        &workspace_size);
printf("%d\n", workspace_size);
cudnnSetConvolutionMathType(conv_descriptor ,CUDNN_TENSOR_OP_MATH);
void * workspace = nullptr;
cudaMalloc(&workspace, workspace_size);

  // allocate the memory on the GPU
  double time1 = (double)clock() / CLOCKS_PER_SEC;
  using std::chrono::system_clock;
  system_clock::time_point today = system_clock::now();

  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_weight, weight_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, out_size * sizeof(C_DATATYPE));

  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaMemcpy(dev_weight, weight, weight_size * sizeof(DATATYPE),
             cudaMemcpyHostToDevice);
  cudaStream_t stream = nullptr;
  for (int i = 0; i < WARMUP; i++) {

  //   const float alpha = 1.0f;
  //   const float beta = 0.0f;
  //   cudnn_state = cudnnConvolutionForward(
  // handle_cudnn,
  // &alpha,
  // input_descriptor,
  // dev_input,
  // kernel_descriptor,
  // dev_weight,
  // conv_descriptor,
  // perfResults[0].algo,
  // workspace,
  // workspace_size,
  // &beta,
  // output_descriptor,
  // dev_out);
  cutlass_nhwc_conv(dev_input, dev_weight, dev_out, batch, ic, ih, iw, kh, kw,
    oc, pad_h, pad_w, stride_h, stride_w, oh, ow, stream);
  }

  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);
  cudaEventRecord(beg);

  for (int i = 0; i < REPEATE; i++) {

    // const float alpha = 1.0f;
    // const float beta = 0.0f;
    // cudnnConvolutionForward(
    // handle_cudnn,
    // &alpha,
    // input_descriptor,
    // dev_input,
    // kernel_descriptor,
    // dev_weight,
    // conv_descriptor,
    // perfResults[0].algo,
    // workspace,
    // workspace_size,
    // &beta,
    // output_descriptor,
    // dev_out);
    cutlass_nhwc_conv(dev_input, dev_weight, dev_out, batch, ic, ih, iw, kh, kw,
      oc, pad_h, pad_w, stride_h, stride_w, oh, ow, stream);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("gpu conv compute time: %f\n", elapsed_time);
  double Gflops =
      REPEATE * ((float)out_size * ic * kh * kw * 2 / 1000000) / elapsed_time;
  printf("Gflops: %5.2f \n", Gflops);

  cudaMemcpy(out_from_dev, dev_out, out_size * sizeof(C_DATATYPE),
             cudaMemcpyDeviceToHost);

  double time2 = (double)clock() / CLOCKS_PER_SEC;
  system_clock::time_point now = system_clock::now();
  auto ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "gpu total time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("gpu total time:%lf\n", double(time2 - time1) * 1000);

  time1 = (double)clock() / CLOCKS_PER_SEC;
  today = system_clock::now();

  // results calculated in cpu is always fp32
  float *out_cpu_fp32 = (float *)malloc(sizeof(float) * out_size);
  memset(out_cpu_fp32, 0, sizeof(float) * out_size);

  naive_conv_cpu(input, weight, out_cpu_fp32, batch, ic, ih, iw, kh, kw, oc,
                 pad_h, pad_w, stride_h, stride_w);

  time2 = (double)clock() / CLOCKS_PER_SEC;
  now = system_clock::now();
  ts = std::chrono::duration_cast<std::chrono::microseconds>(now - today);
  std::cout << "cpu time:" << ts.count() / 1000.0 << "ms" << std::endl;
  printf("cpu time:%lf\n", double(time2 - time1) * 1000);

  printf("max_diff: %f\n", diff(out_from_dev, out_cpu_fp32, out_size));

  cudaDeviceReset();
  cudaFreeHost(input);
  cudaFreeHost(weight);
  cudaFreeHost(out_from_dev);
  free(out_cpu_fp32);
  return 0;
}
