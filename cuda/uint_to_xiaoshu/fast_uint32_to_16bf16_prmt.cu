#include <stdint.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <cuda_fp16.h>
// nvcc fast_uint32_to_16bf16_prmt.cu -arch sm_80

__device__ inline void int32_to_16bf16(
    __nv_bfloat16 halves[16], uint32_t signed_chars) {
  uint32_t* h = reinterpret_cast<uint32_t*>(halves);
  uint32_t i8s = signed_chars;

  static constexpr uint32_t mask_for_elt_01 = 0x5150;
  static constexpr uint32_t mask_for_elt_23 = 0x5352;
  static constexpr uint32_t start_byte_for_fp16 = 0x43434343;


  uint32_t i8s_0 = i8s & 0x03030303;
  uint32_t i8s_1 = (i8s & 0x0C0C0C0C) >> 2;
  uint32_t i8s_2 = (i8s & 0x30303030) >> 4;
  uint32_t i8s_3 = (i8s & 0xC0C0C0C0) >> 6;

  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[0])
               : "r"(i8s_0), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[1])
               : "r"(i8s_0), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[2])
               : "r"(i8s_1), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[3])
               : "r"(i8s_1), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[4])
               : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[5])
               : "r"(i8s_2), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[6])
               : "r"(i8s_3), "n"(start_byte_for_fp16), "n"(mask_for_elt_01));
  asm volatile("prmt.b32 %0,%1,%2,%3;\n"
               : "=r"(h[7])
               : "r"(i8s_3), "n"(start_byte_for_fp16), "n"(mask_for_elt_23));

  
  __nv_bfloat162* h2 = reinterpret_cast<__nv_bfloat162*>(halves);
  __nv_bfloat162 tmp = {128.f,128.f};
  h2[0] = h2[0] - tmp;
  h2[1] = h2[1] - tmp;
  h2[2] = h2[2] - tmp;
  h2[3] = h2[3] - tmp;
  h2[4] = h2[4] - tmp;
  h2[5] = h2[5] - tmp;
  h2[6] = h2[6] - tmp;
  h2[7] = h2[7] - tmp;
}


__global__ void helloFromGPU (void)
{
    const int N = 16;
    __nv_bfloat16 halves[N];
    uint32_t uint32_weight[N] = {0,3,2,1, 0,1,2,3, 3,0,2,1, 3,2,0,1};

    uint32_t weight = 0;
    for (int i = 0; i < N; ++i) {
        weight |= (uint32_weight[i] << i*(32/N));
    }

    int32_to_16bf16(halves, weight);

    for (int i = 0; i < N; ++i) {
        printf("%f\n", (float)(halves[i]));
    }
}

int main(void)
{
helloFromGPU <<<1, 1>>>();
cudaDeviceReset();
return 0;
}
