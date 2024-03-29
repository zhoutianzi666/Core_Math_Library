


#include "utility.h"
using DATATYPE = half;
#include "cutlass/util/device_nchw_to_nhwc.h"
//#include "cutlass/util/device_nhwc_to_nchw.h"


template <typename T>
__global__ void batch_transpose_kernel(
    T* output, const T* input, const int batch, const int M, const int N) {
  const int num = M * N;
  // "+1" to avoid smem bank conflict
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid = tid / 32;
  const int32_t lid = tid % 32;
  const int32_t batch_i = blockIdx.z;
  const int32_t mi0 = blockIdx.y * 32;
  const int32_t ni0 = blockIdx.x * 32;

  const size_t input_idx = batch_i * num + (mi0 + wid) * N + ni0;
  const T* A = input + input_idx;
  if (ni0 + lid < N) {
    const int lid_x_33 = lid * 33;
    if ((mi0 + 32) <= M) {
      int mi = wid;  // between 0 and 7
#pragma unroll
      for (int mLoopIdx = 0; mLoopIdx < 4; mLoopIdx++) {
        shbuf[lid_x_33 + mi] = A[lid];
        A = &A[8 * N];
        mi += 8;
      }
    } else {
      for (int mi = wid; mi < 32; mi += 8) {
        if ((mi + mi0) < M) {
          shbuf[lid_x_33 + mi] = A[lid];
        }
        A = &A[8 * N];
      }
    }
  }
  __syncthreads();

  const int32_t miOut = mi0 + lid;
  output = &output[batch_i * num + miOut];
  // 下面是将数据从smem中转移到global memory中
  if (miOut < M) {
    if (ni0 + 32 < N) {
      int nI = wid;
#pragma unroll
      for (int nLoopIdx = 0; nLoopIdx < 4; ++nLoopIdx) {
        output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
        nI += 8;
      }
    } else {
      for (int nI = wid; nI < 32; nI += 8) {
        if (ni0 + nI < N) {
          output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
        }
      }
    }
  }
}

void cutlass_nhwc_nchw(const DATATYPE *input, DATATYPE *output, int batch, int ic,
    int ih, int iw) {

// cutlass::Tensor4DCoord a = cutlass::Tensor4DCoord(batch, ih, iw, ic);
// cutlass::Tensor4DCoord b = cutlass::Tensor4DCoord(batch, iw, ic, ih);
// cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> c {(cutlass::half_t *)input, cutlass::make_Coord(iw, ih * iw, ic * ih * iw)};
// cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNCHW> d {(cutlass::half_t *)output, cutlass::make_Coord(iw, iw * ih, ic * iw * ih)};

// cutlass::nhwc_to_nchw(a, b, c, d, nullptr);


dim3 grid((ic + 31)/32, (ih*iw + 31)/32, batch);
dim3 block(32, 8);
batch_transpose_kernel<<<grid, block, 0>>>(output, input, 
                                                batch, ih * iw, ic);

}




// 下面这个尝试在上面的基础上进行向量化！
template <typename T>
__global__ void batch_transpose_kernel_vec(
    T* output, const T* input, const int batch, const int M, const int N) {
  const int num = M * N;
  // "+1" to avoid smem bank conflict
  __shared__ T shbuf[32 * (32 + 1)];
  const int32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int32_t wid = tid / 32;
  const int32_t lid = tid % 32;
  const int32_t batch_i = blockIdx.z;
  const int32_t mi0 = blockIdx.y * 32;
  const int32_t ni0 = blockIdx.x * 32;

  const size_t input_idx = batch_i * num + (mi0 + wid) * N + ni0;
  const T* A = input + input_idx;
  if (ni0 + lid < N) {
    const int lid_x_33 = lid * 33;
    if ((mi0 + 32) <= M) {
      int mi = wid;  // between 0 and 7
#pragma unroll
      for (int mLoopIdx = 0; mLoopIdx < 4; mLoopIdx++) {
        shbuf[lid_x_33 + mi] = A[lid];
        A = &A[8 * N];
        mi += 8;
      }
    } else {
      for (int mi = wid; mi < 32; mi += 8) {
        if ((mi + mi0) < M) {
          shbuf[lid_x_33 + mi] = A[lid];
        }
        A = &A[8 * N];
      }
    }
  }
  __syncthreads();

  const int32_t miOut = mi0 + lid;
  //output = &output[batch_i * num + miOut];
  // 下面是将数据从smem中转移到global memory中
  // 
  if (miOut < M) {
    if (ni0 + 32 < N || 1) {
      int nI = wid;
#pragma unroll
      // for (int nLoopIdx = 0; nLoopIdx < 4; ++nLoopIdx) {
      //   output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
      //   nI += 8;
      // }

      int threaed_id_in_block = wid * 32 + lid;
      int new_row = tid / 8; // (0-8之间)
      int new_col = tid % 8;   //
      half tmp[4];
      tmp[0] = shbuf[(new_row)*33 + new_col*4+0];
      tmp[1] = shbuf[(new_row)*33 + new_col*4+1];
      tmp[2] = shbuf[(new_row)*33 + new_col*4+2];
      tmp[3] = shbuf[(new_row)*33 + new_col*4+3];

     *(float2*)(&(output[(ni0 + new_row) * M + new_col*4 + mi0])) = *((float2*)tmp);

    } else {
      for (int nI = wid; nI < 32; nI += 8) {
        if (ni0 + nI < N) {
          output[(ni0 + nI) * M] = shbuf[(nI)*33 + lid];
        }
      }
    }
  }
}

void cutlass_nchw_nhwc(const DATATYPE *input, DATATYPE *output, int batch, int ic,
                  int ih, int iw) {
// cutlass::Tensor4DCoord a = cutlass::Tensor4DCoord(batch, iw, ic, ih);
// cutlass::Tensor4DCoord b = cutlass::Tensor4DCoord(batch, ih, iw, ic);
// cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNCHW> c {(cutlass::half_t *)input, cutlass::make_Coord(iw, ih * iw, ic * ih * iw)};
// cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> d {(cutlass::half_t *)output, cutlass::make_Coord(iw, iw * ih, ic * iw * ih)};

//cutlass::nchw_to_nhwc(a, b, c, d, nullptr);

// 一定要去记住第0维度是宽度方向啊！
dim3 grid((ih*iw + 31)/32, (ic + 31)/32, batch);
dim3 block(32, 8);
batch_transpose_kernel<<<grid, block, 0>>>(output, input, 
                                                batch, ic, ih * iw);

}