#pragma once
#include <stdio.h>
#include <cudnn_v8.h>
#include <string>
#include <iostream>

#include "cublas_v2.h"
#include "cutlass/gemm/device/gemm.h"
#include "utility.h"

using C_DATATYPE = half;

using DATATYPE = half;
#include <algorithm>
#define warp_M 16
#define warp_N 8
#define warp_K 8
#define WARP_SIZE 32
using DATATYPE = half;

void CUDNN_CHECK(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    printf("CUDNN 不能实施\n");
  }
}


void cudnn_nhwc_conv(ConvAllParams params) {

  auto handle_cudnn = params.handle_cudnn;
  int batch = params.batch;
  int ih = params.ih;
  int iw = params.iw;
  int ic = params.ic;
  int oc = params.oc;
  int kh = params.kh;
  int kw = params.kw;
  int pad_h0 = params.pad_h0;
  int pad_h1 = params.pad_h1;
  int pad_w0 = params.pad_w0;
  int pad_w1 = params.pad_w1;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

  int oh = params.oh;
  int ow = params.ow;
  auto input = params.input;
  auto weight = params.weight;
  auto bias = params.bias;
  auto output = params.output;

  int groups = params.groups;
  int kc = ic / groups;

  int out_size = batch * oc * oh * ow;

  size_t cudnn_workspace_size = params.cudnn_workspace_size;
  void *cudnn_workspace = params.cudnn_workspace;

  cudnnTensorDescriptor_t input_descriptor;
  auto cudnn_layout = CUDNN_TENSOR_NHWC;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_descriptor, cudnn_layout,
                                         CUDNN_DATA_HALF, batch, ic, ih, iw));

  cudnnTensorDescriptor_t output_descriptor;
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_descriptor));
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_descriptor, cudnn_layout,
                                         CUDNN_DATA_HALF, batch, oc, oh, ow));
  cudnnFilterDescriptor_t kernel_descriptor;
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_descriptor));
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_HALF,
                                         cudnn_layout, oc, kc, kh, kw));

  cudnnConvolutionDescriptor_t conv_descriptor;
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_descriptor));
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_descriptor, pad_h0, pad_w0,  // 右边不需要告诉他,还是默认是对称的呢？
      stride_h, stride_w,               // stride
      dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_descriptor, groups));
  cudnnSetConvolutionMathType(conv_descriptor, CUDNN_TENSOR_OP_MATH);

//   std::cout << "这个我猜可能是逻辑判断来确定的最优算法！" << std::endl;
//   int returnedAlgoCount;
//  int requestedAlgoCount = 100;
//   cudnnConvolutionFwdAlgoPerf_t perfResults[100];

//   CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
//       handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
//       output_descriptor, requestedAlgoCount, &returnedAlgoCount, perfResults));


//   printf("\t返回的算法个数是：%d\n", returnedAlgoCount);

//   for (int i = 0; i < returnedAlgoCount; i++) {
//     std::cout << "\t perfResults[" << i<< "]:" << cudnnAlgoName(perfResults[i].algo) 
//             <<" " << perfResults[i].status
//             << std::endl;
//   }

// 我认为这个是不需要的，workspace我们预先给他吧！
//   CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
//       handle_cudnn, input_descriptor, kernel_descriptor, conv_descriptor,
//       output_descriptor, perfResults[0].algo, &cudnn_workspace_size));


//   // ---------------------------this is also cuDNN-----------------------------------------------
//   std::cout << "这个我猜可能是真实的计算来确定的最优算法！" << std::endl;
//   CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
//       handle_cudnn, input_descriptor, input, kernel_descriptor, weight,
//       conv_descriptor, output_descriptor, output, 100, &returnedAlgoCount,
//       perfResults, cudnn_workspace, cudnn_workspace_size));
//   for (int i = 0; i < returnedAlgoCount; i++) {
//   std::cout << "\tperfResults[" << i<< "]:" << cudnnAlgoName(perfResults[i].algo) 
//             <<" " << perfResults[i].status
//             << std::endl;
//   }
//   printf("\t返回的算法个数是：%d\n", returnedAlgoCount);
//   cudaMemset(output, 0, sizeof(C_DATATYPE) * out_size);

  // 上面必须要清零！因为cuDNN要计算这个结果！

  // ------------------------cuDNN ends-----------------------------------------------------

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
    handle_cudnn,
    &alpha, 
    input_descriptor, input,
    kernel_descriptor, weight,
    conv_descriptor, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, cudnn_workspace, cudnn_workspace_size, &beta, 
    output_descriptor, output));
}




std::string cudnnAlgoName(cudnnConvolutionFwdAlgo_t algo)
{
    switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
         return std::string("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM");
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
        return std::string("CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM");
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
        return std::string("CUDNN_CONVOLUTION_FWD_ALGO_GEMM");
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
        return std::string("CUDNN_CONVOLUTION_FWD_ALGO_DIRECT");
        
    default:
        return std::string("");
    }
    return std::string("");
}

