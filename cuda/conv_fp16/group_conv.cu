// #pragma once
// #include <stdio.h>

// #include <iostream>

// #include "cublas_v2.h"
// #include "cutlass/gemm/device/gemm.h"
// #include "utility.h"

// #include <algorithm>

// #include "cutlass/cutlass.h"
// #include "cutlass/conv/kernel/default_conv2d_group_fprop.h"
// #include "cutlass/conv/device/implicit_gemm_convolution.h"
// #include "cutlass/conv/kernel/default_conv2d_fprop.h"
// #include "cutlass/epilogue/thread/linear_combination_silu.h"


// using ElementAccumulator = float;
// using ElementComputeEpilogue = float;
// using ElementInputA = cutlass::half_t;
// using ElementInputB = cutlass::half_t;
// using ElementOutput = cutlass::half_t;     
// using LayoutInputA = cutlass::layout::TensorNHWC;
// using LayoutInputB = cutlass::layout::TensorNHWC;
// using LayoutOutput = cutlass::layout::TensorNHWC;

// using MMAOp = cutlass::arch::OpClassTensorOp;

// using SmArch = cutlass::arch::Sm75;

// using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;   // Threadblock tile shape
// using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;          // Warp tile shape
// using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;    // TensorCore instruction shape

// // This code section describes how threadblocks are scheduled on GPU
// using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;

// // Number of pipelines you want to use
// constexpr int NumStages = 2;

// // This code section describes the epilogue part of the kernel, we use default value
// using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
//     ElementOutput,                                     // Data type of output matrix.
//     128 / cutlass::sizeof_bits<ElementOutput>::value,  // The number of elements per vectorized.
//                                                        // memory access. This becomes the vector width of
//                                                        // math instructions in the epilogue too.
//     ElementAccumulator,                                // Data type of accumulator
//     ElementComputeEpilogue>;                           // Data type for alpha/beta in linear combination


// // Optimized kernel and operation for single group problem size
// using OptimizedSingleGroupKernel = typename cutlass::conv::kernel::DefaultConv2dGroupFprop<
//   ElementInputA, LayoutInputA,
//   ElementInputB, LayoutInputB,
//   ElementOutput, LayoutOutput,
//   ElementAccumulator,
//   MMAOp,
//   SmArch,
//   ThreadblockShape,
//   WarpShape,
//   InstructionShape,
//   EpilogueOp,
//   SwizzleThreadBlock,
//   NumStages,
//   cutlass::arch::OpMultiplyAdd,
//   cutlass::conv::GroupMode::kSingleGroup,
//   cutlass::conv::IteratorAlgorithm::kOptimized
// >::Kernel;
// using ImplicitGemm = cutlass::conv::device::ImplicitGemmConvolution<OptimizedSingleGroupKernel>;

// void cutlass_group_conv_bias_swish(ConvAllParams params) {
//   cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;

//   int batch = params.batch;
//   int ih = params.ih;
//   int iw = params.iw;
//   int ic = params.ic;
//   int oc = params.oc;
//   int kh = params.kh;
//   int kw = params.kw;
//   int pad_h0 = params.pad_h0;
//   int pad_h1 = params.pad_h1;
//   int pad_w0 = params.pad_w0;
//   int pad_w1 = params.pad_w1;
//   int stride_h = params.stride_h;
//   int stride_w = params.stride_w;
//   int dilation_h = params.dilation_h;
//   int dilation_w = params.dilation_w;

//   int oh = params.oh;
//   int ow = params.ow;
//   auto input = params.input;
//   auto weight = params.weight;
//   auto bias = params.bias;
//   auto output = params.output;

//   int groups = params.groups;
//   int kc = ic / groups;

//   cutlass::conv::Conv2dProblemSize problem_size(
//       {batch, ih, iw, ic},  {oc, kh, kw, kc}, {pad_h0, -1, pad_w0, -1},
//       {stride_h, stride_w}, {dilation_h, dilation_w}, {batch, oh, ow, oc}, mode,
//       1, groups);

//   typename ImplicitGemm::Arguments arguments{
//       problem_size,
//       {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},
//       {(cutlass::half_t *)weight, {kc, kc * kw, kc * kw * kh}},
//       {(cutlass::half_t *)bias, {0, 0, 0}},
//       {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},
//       {1.f, 1.f},
//       cutlass::conv::SplitKMode::kSerial};
//       // cutlass::conv::SplitKMode::kParallel 也可以用啊，但是啥时候会快呢？

//   ImplicitGemm implicit_gemm_op;
//   size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
//   // 其实bytes基本都是0，因为是kSerial。
//   assert(bytes == 0);
//   void *workspace;
//   //cudaMalloc(&workspace, bytes);

//   cutlass::Status status = implicit_gemm_op.can_implement(arguments);
//   CUTLASS_CHECK(status);
//   status = implicit_gemm_op.initialize(arguments, workspace);
//   CUTLASS_CHECK(status);
//   status = implicit_gemm_op();
//   CUTLASS_CHECK(status);
//   // cudaFree还蛮费时间的！
//   //cudaFree(&workspace, bytes);
// }
