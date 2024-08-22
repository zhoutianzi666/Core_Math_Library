





#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>
#include <cuda_fp16.h>

using Element = half;

__global__ void kernel1(__restrict__ Element *y, int N, __restrict__ Element *x) {
    int num = 8;
    int global_idx = (blockIdx.x * blockDim.x + threadIdx.x) * num;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x * num) {
        float4 tmp = *reinterpret_cast<float4 *>(x + i);
        tmp.x = tmp.y + tmp.x;
        tmp.x = tmp.x + tmp.z;
        tmp.x = tmp.x + tmp.z;
        tmp.x = tmp.y + tmp.w;
        *(float4*)(y+i) = tmp;
    }
}

int main(void) {
  int M = 2;
  int N = 436224;
  int K = 1536;

  thrust::host_vector<Element> h_A(M*K);
  thrust::host_vector<Element> h_B(K*N);
  thrust::host_vector<Element> h_C(K*N);

  for (size_t i = 0; i < h_A.size(); ++i) {
    h_A[i] = static_cast<Element>(1);
  }

//   for (size_t i = 0; i < h_B.size(); ++i) {
//     h_B[i] = static_cast<Element>(1);
//   }

  thrust::device_vector<Element> d_A = h_A;
  thrust::device_vector<Element> d_B = h_B;
  thrust::device_vector<Element> d_C = h_C;

  thrust::device_vector<Element> d_C2 = h_C;

  std::cout << d_C.size() << std::endl;



  constexpr int WARMUP =  10;
  constexpr int REPEATE =  100;

  cudaEvent_t beg, end;

  for (int i = 0; i < WARMUP + REPEATE; i++) {

    if (i == WARMUP) {
      cudaEventCreate(&beg);
      cudaEventCreate(&end);
      cudaEventRecord(beg);
    }
    
    kernel1<<<1080, 1024>>>(thrust::raw_pointer_cast(d_C2.data()), 
                        h_C.size(), 
                        thrust::raw_pointer_cast(d_C.data()));

  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, beg, end);
  printf("copy kernel time: %f us \n", elapsed_time / REPEATE * 1000);
  
 // cudaDeviceReset();
  return 0;
}











