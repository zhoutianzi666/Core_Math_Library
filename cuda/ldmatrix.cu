#include <stdio.h>
#include <iostream>
// nvcc ldmatrix.cu -arch=compute_80 -code=sm_80 && ./a.out 
__global__ void helloFromGPU (void) {
  __shared__ uint32_t aTile[200];

  int tidx = threadIdx.x + blockDim.x * threadIdx.y;
  if (tidx == 0) {
    for (int i = 0; i < 200; ++i) {
        aTile[i] = i;
    }
  }
  __syncthreads();

  int realative_index = tidx % 16 * 8 + tidx / 16 * 4;
  uint32_t a[4];
  uint32_t smem = __cvta_generic_to_shared(aTile+realative_index);
  asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
  : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) 
  : "r"(smem)
  );

  if (tidx == 1) {
    printf("%d \n", (a[0]));
    printf("%d \n", (a[1]));
    printf("%d \n", (a[2]));
    printf("%d \n", (a[3]));
  }
}

int main(void) {
uint3 grid = {1,1,1};
uint3 block = {32,1,1};
helloFromGPU <<<grid, block>>>();
cudaDeviceReset();
return 0;
}
