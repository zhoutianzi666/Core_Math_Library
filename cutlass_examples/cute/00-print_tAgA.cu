#include <cuda.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

using namespace cute;

template <typename T, typename G2SCopy, int M, int N>
__global__ void copy_global_shm_register(const T *Aptr) {
    int idx = threadIdx.x;
    auto gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_thread_slice(idx);
    auto tAgA = g2s_thr_copy.partition_S(gA);

    if (idx == 0) {
        PRINT("gA.shape()", gA.shape());
        PRINT("gA.stride()", gA.stride());
        PRINT("tAgA.shape()", tAgA.shape());
        PRINT("tAgA.stride()", tAgA.stride());

        for (int i = 0; i < 8; ++i) {
            float b0 = (float)(tAgA((0,i),0,0));
            float b1 = (float)(tAgA((i,0),0,0));
            const T* b2 = tAgA((_,_),0,0).data().get();
            printf("%f\n", b0);
            printf("%f\n", b1);
            printf("%f\n", (float)(b2[i]));
        }
    }
}


int main() {
    using T = cute::half_t;
    
    constexpr int M = 128;
    constexpr int N = 64;

    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using G2SCopy =
        decltype(make_tiled_copy(g2s_copy_atom{},
                                 make_layout(make_shape(Int<32>{}, Int<4>{}),
                                             make_stride(Int<4>{}, Int<1>{})),
                                 make_layout(make_shape(Int<1>{}, Int<8>{}), 
                                             make_stride(Int<8>{}, Int<1>{}))));

    thrust::host_vector<T> h_A(M*N);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = i;
    }
    thrust::device_vector<T> d_A = h_A;
    auto Aptr = thrust::raw_pointer_cast(d_A.data());

    dim3 block(32*4);
    copy_global_shm_register<T, G2SCopy, M, N><<<1, block>>>(Aptr);
}
