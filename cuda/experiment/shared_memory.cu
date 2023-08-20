#include <stdio.h>
#include <cuda_fp16.h>
using DATATYPE = float;

__global__ void helloFromGPU (DATATYPE* in, int n)
{
    __shared__ DATATYPE aTile[32][32];
    DATATYPE sum = 1;
    for (int i = 0; i < 32; i++)
    { 
      // 这种访问方式导致这段代码被循环展开了，并且被向量循环展开了！
      sum = sum + aTile[threadIdx.x][i];
    }
    *in = sum;
}

int main(void)
{

int n = 51;
DATATYPE* a = (DATATYPE*)malloc(sizeof(DATATYPE)* n);
for(int i = 0;i < n;i ++)
  a[i] = (half)((rand()%9999)/10000.0);

DATATYPE *dev_a;
cudaMalloc( (void**)&dev_a, n * sizeof(DATATYPE) );
cudaMemcpy(dev_a, a, n * sizeof(DATATYPE), cudaMemcpyHostToDevice );

helloFromGPU <<<1, 32>>>(dev_a, n);

cudaDeviceReset();
free(a);
cudaFree(dev_a);
cudaDeviceReset();
return 0;
}
