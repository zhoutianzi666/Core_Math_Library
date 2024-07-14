#include <stdint.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cuda_fp16.h>
// nvcc fast_uint32_to_4fp16.cu -arch sm_80

// Specialization for fast cast from FP16 -> int8
__device__ inline void fast_cvt_4_packed_signed_i8s_to_2_half2s(
    half halves[4], uint32_t signed_chars) {
  uint32_t* h = reinterpret_cast<uint32_t*>(halves);
  uint32_t i8s = signed_chars;

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x64646464;
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[0])
               : "r"(h[0]), "r"(I8s_TO_F16s_MAGIC_NUM));
  asm volatile("sub.f16x2 %0, %1, %2;\n"
               : "=r"(h[1])
               : "r"(h[1]), "r"(I8s_TO_F16s_MAGIC_NUM));
}




__global__ void helloFromGPU (void)
{

    const int N = 4;
    half halves[N];
    int8_t int_weight[N] = {1,13,24,-12};
    uint8_t uint_weight[N];
    for (int i = 0; i < N; ++i) {
        uint_weight[i] = int_weight[i] + 128;
    }

    uint32_t weight = *(uint32_t*)(uint_weight);
    
    fast_cvt_4_packed_signed_i8s_to_2_half2s(halves, weight);

    for (int i = 0; i < N; ++i) {
        printf("%f\n", (float)(halves[i]));
    }
}

int main(void)
{
helloFromGPU <<<1, 1>>>();
cudaDeviceReset();

// cudaEvent_t beg, end;
// cudaEventCreate(&beg);
// cudaEventCreate(&end);

// cudaEventRecord(beg);
// helloFromGPU <<<1, 1>>>();
// cudaEventRecord(end);

// cudaEventSynchronize(end);
// float elapsed_time;
// cudaEventElapsedTime(&elapsed_time, beg, end);
// printf("gpu conv compute time: %f\n", elapsed_time);
// cudaDeviceReset();
return 0;
}

