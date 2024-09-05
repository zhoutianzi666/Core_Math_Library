#include <cuda.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

// https://github.com/NVIDIA/cutlass/issues/1746

#define PRINT(name, content) \
    print(name);             \
    print(" : ");            \
    print(content);          \
    print("\n");

using namespace cute;

template <typename T, typename G2SCopy, typename SmemLayout, int M, int N>
__global__ void copy_global_shm_register(const T *Aptr)
{
    int idx = threadIdx.x;
    extern __shared__ T shm_data[];
    T *Ashm = shm_data;
    
    auto gA = make_tensor(make_gmem_ptr(Aptr), make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{}));
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayout{});

    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_thread_slice(idx);

    auto tAgA = g2s_thr_copy.partition_S(gA);
    auto tAsA = g2s_thr_copy.partition_D(sA);
    // tAgA 是 global memory地址
    // tAsA 是 shared memory地址，但是每个thread指向哪个地址呢？这个是比较有讲究的！
    cute::copy(g2s_tiled_copy, tAgA((_,_),_,_), tAsA((_,_),_,_));
    __syncthreads();

    if (idx == 9) {
        PRINT("gA.shape()", gA.shape());
        PRINT("gA.stride()", gA.stride());
        PRINT("tAgA.shape()", tAgA.shape());
        PRINT("tAgA.stride()", tAgA.stride());

        PRINT("tAsA.shape()", tAsA.shape());
        PRINT("tAsA.stride()", tAsA.stride());
        
        const T* p = (const T*)(tAsA((_,_),0,0).data().get());
        printf("%d\n", p - Ashm);
        for (int i = 0; i < 8; ++i) {
            float a = (float)(tAsA((0,i),0,0));
            float b = (float)(tAgA((0,i),0,0));
            if (a!=b) printf("errors\n\n");
        }
    }
}


int main()
{
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
                                 make_layout(make_shape(Int<1>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}))));
    
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<32>{}),
                    make_stride(Int<32>{}, Int<1>{}))));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              make_shape(Int<M>{}, Int<N>{})));

    static constexpr int shm_size = cute::cosize(SmemLayout{}) * sizeof(T);

    PRINT("SmemLayout", SmemLayout{}.shape());
    PRINT("SmemLayout", SmemLayout{});

    thrust::host_vector<T> h_A(M*N);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = i;
    }
    thrust::device_vector<T> d_A = h_A;
    auto Aptr = thrust::raw_pointer_cast(d_A.data());

    dim3 block(32*4);
    copy_global_shm_register<T, G2SCopy, SmemLayout, M, N><<<1, block, shm_size>>>(Aptr);
}
