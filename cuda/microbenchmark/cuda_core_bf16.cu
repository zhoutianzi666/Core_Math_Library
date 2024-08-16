#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define dtype __nv_bfloat162

#define flops_each_thread 40000
// nvcc cuda_core_bf16.cu -arch sm_80 -o a.out
// cuobjdump -ptx a.out

__global__ void kernel(const dtype *x, dtype *y, int N) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = global_idx; i < N; i += blockDim.x * gridDim.x) {
        dtype a = x[i];
        dtype b;

        unsigned bb = *(unsigned*)(&b);
        unsigned aa = *(unsigned*)(&a);

        for (int ii = 0; ii < flops_each_thread; ii++) {

            asm ("fma.rn.bf16x2 %0, %1, %2, %3;\n" 
                : "=r"(bb) 
                : "r"(aa), "r"(bb), "r"(bb)
            );

        }

        b = *(dtype*)(&bb);
        y[i] = b;
    }
}

int main() {
    const int WARMUP_ITER = 10;
    const int BENCH_ITER = 50;
    const int N_DATA = 1024 * 1024 * 128;
    dtype *x, *y;
    cudaMalloc(&x, N_DATA * sizeof(dtype));
    cudaMalloc(&y, N_DATA * sizeof(dtype));
    cudaMemset(x, 0, N_DATA * sizeof(dtype));
    int blocksize = 32;
    int grid = 1024*1024;
    cudaEvent_t start, stop;

    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<<<grid, blocksize>>>(x, y, N_DATA);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int i = 0; i < BENCH_ITER ; ++i) {
        kernel<<<grid, blocksize>>>(x, y, N_DATA);
    }
    cudaEventRecord(stop);

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    printf("time: %f ms\n", time_ms);
    
    float flops = (N_DATA * (float)(flops_each_thread) * 2 * BENCH_ITER) / (time_ms) * 1000;
    flops = flops * 2; // 因为是bfloat16_2！
    printf("flops %f TFlops\n", flops/1000000000000 );
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(x);
    cudaFree(y);
    return 0;
}
