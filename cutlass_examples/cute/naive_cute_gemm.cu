#include <cuda.h>
#include <stdlib.h>
#include <iostream>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>


#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

#define PRINTTENSOR(name, content) \
    print(name);                   \
    print(" : ");                  \
    print_tensor(content);         \
    print("\n");
    



using namespace cute;

template <typename T, typename MMA, int M, int N, int K>
__global__ void mma_simple(T *Cptr, const T *Aptr, const T *Bptr)
{
    MMA tiled_mma;

    // ThrMMA
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);

    // tensor layout should be in static
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<M>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, Int<1>{}));
    Tensor C = make_tensor(make_gmem_ptr(Cptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    auto tAgA = thr_mma.partition_A(A);
    auto tBgB = thr_mma.partition_B(B);
    auto tCgC = thr_mma.partition_C(C);

    auto tArA = thr_mma.partition_fragment_A(A);
    auto tBrB = thr_mma.partition_fragment_B(B);
    auto tCrC = thr_mma.partition_fragment_C(C);

    if (threadIdx.x == 0) {
        // MMA 由MMA指令决定，不受MMAThrLayout和MMAValLayout影响
        // A,B,C 对应为： 16*16/32=8=(2,2,2), 16*8/32=4=(2,2), 16*8/32=4=(2,2)

        // MMA_M, MMA_K, MMA_N 由MMA指令、MMAThrLayout和源Tensor shape决定，不受MMAValLayout影响
        // MMA_M = M / (mma_op_m * thr_layout_m)
        // MMA_N = N / (mma_op_n * thr_layout_n)
        // MMA_K = K / (mma_op_k * thr_layout_k)

        // (MMA, MMA_M, MMA_K)
        PRINT("tArA.shape", tArA.shape());
        // (MMA, MMA_N, MMA_K)
        PRINT("tBrB.shape", tBrB.shape());
        // (MMA, MMA_M, MMA_N)
        PRINT("tCrC.shape", tCrC.shape());
    }

    cute::copy(tAgA, tArA);
    cute::copy(tBgB, tBrB);
    clear(tCrC);

    // cute::gemm, warp level
    // 语义：处理tCrC, tArA, tBrB 对应的partition前tensor A,B,C的 C= A*B+C, 内部会拆解成大量的mma_atom指令
    cute::gemm(tiled_mma, tCrC, tArA, tBrB, tCrC);

    // // thread level
    cute::copy(tCrC, tCgC);
}

/*
Profile
thr_layout,val_layout
thr_layout比较影响速度


M=128, N=128, K=32
Shape<_1, _1, _1>  Shape<_1, _1, _1>
0.040183ms

Shape<_2, _2, _2>  Shape<_1, _1, _1>
0.037417ms

Shape<_4, _4, _2>  Shape<_1, _1, _1>
0.0368947ms

Shape<_4, _4, _2>  Shape<_2, _1, _1>
0.0366608ms

Shape<_4, _4, _2>  Shape<_2, _4, _1>
0.0392909ms

------------------------------------------

M=256, N=256, K=128
Shape<_1, _1, _1>  Shape<_1, _1, _1>
0.751002ms

Shape<_2, _2, _2>  Shape<_1, _1, _1>
0.251904ms

Shape<_4, _4, _2>  Shape<_1, _1, _1>
0.214118ms

Shape<_2, _4, _4>  Shape<_1, _1, _1>
0.206029ms

Shape<_2, _4, _4>  Shape<_2, _2, _2>
0.205395ms

Shape<_2, _4, _4>  Shape<_4, _4, _4>
0.239418ms

*/

int main() {

    cudaEvent_t start, end;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    using T = cute::half_t;
    // MMAOperation, M=16, N=8, K=16, type=half
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    // MMA_Traits
    using mma_traits = MMA_Traits<mma_op>;
    // MMA_ATOM
    using mma_atom = MMA_Atom<mma_traits>;
    // TiledMMA
    // using MMA = decltype(make_tiled_mma(mma_atom{},
    //                                     make_layout(Shape<_1, _1, _1>{}),   // thr_layout
    //                                     make_layout(Shape<_1, _1, _1>{}))); // val_layout

    using MMA = decltype(make_tiled_mma(mma_atom{},
                                        //make_layout(Shape<_2, _2, _4>{}),   // thr_layout
                                        make_layout(Shape<_1, _1, _1>{}),   // thr_layout
                                        make_layout(Shape<_1, _1, _1>{}))); // val_layout
    // constexpr int M = 16;
    // constexpr int N = 16;
    // constexpr int K = 16;

    // constexpr int M = 16;
    // constexpr int N = 8;
    // constexpr int K = 16;

    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 128;

    thrust::host_vector<T> h_A(M*K);
    thrust::host_vector<T> h_B(K*N);
    thrust::host_vector<T> h_C(M*N);

    for (int i = 0; i < h_A.size(); ++i) {
        h_A[i] = (rand() % 100) / 10.f;
    }
    for (int i = 0; i < h_B.size(); ++i) {
        h_B[i] = (rand() % 100) / 10.f;
    }
    for (int i = 0; i < h_C.size(); ++i) {
        h_C[i] = (rand() % 100) / 10.f;
    }

    thrust::device_vector<T> d_A = h_A;
    thrust::device_vector<T> d_B = h_B;
    thrust::device_vector<T> d_C = h_C;
    auto Aptr = thrust::raw_pointer_cast(d_A.data());
    auto Bptr = thrust::raw_pointer_cast(d_B.data());
    auto Cptr = thrust::raw_pointer_cast(d_C.data());

    dim3 block(size(MMA{}));
    print(size(MMA{}));
    print("\n");
    cudaEventRecord(start);

    int count = 10;
    for (int i = 0; i < count; ++i) {
        mma_simple<T, MMA, M, N, K><<<1, block>>>(Cptr, Aptr, Bptr);
    }

    h_C = d_C;

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; ++k) {
                // h_B是col major的！
                sum += (float)(h_A[i * K + k]) * (float)(h_B[k + j * K]);
            }
            float data = h_C[i * N + j];
            printf("%f\n", sum);
            printf("%f\n", data);
        }
    }

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsedTime, start, end);
    std::cout << "mma_simple took " << elapsedTime / count << "ms." << std::endl;
}