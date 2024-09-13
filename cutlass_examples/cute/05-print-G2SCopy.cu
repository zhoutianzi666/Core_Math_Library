#include <cuda.h>
#include <stdlib.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

using namespace cute;

template <typename T, typename G2SCopy, int M, int N>
__global__ void copy_global_shm_register(const T *Aptr)
{
    int idx = threadIdx.x;
    auto gA = make_tensor(make_gmem_ptr(Aptr), 
                          make_shape(Int<M>{}, Int<N>{}), 
                          make_stride(Int<N>{}, Int<1>{}));
    
    G2SCopy g2s_tiled_copy;
    auto g2s_thr_copy = g2s_tiled_copy.get_thread_slice(idx);

    auto tAgA = g2s_thr_copy.partition_S(gA);



    if (idx == 0) {
        // print_tensor(gA);
        print_tensor(tAgA);
        print(tAgA.shape());
    }
}


int main() {
    using T = cute::half_t;
    
    constexpr int M = 64;
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
    
    // 打印这个global到shared memory的copy操作，这个操作是指定了线程的行为！还不知道输入的layout！
    if (0) {
        print_latex(G2SCopy{});
        return 0;
    }

    thrust::host_vector<T> h_A(M*N);
    for (int i = 0; i < M * N; ++i) {
        h_A[i] = i;
    }
    thrust::device_vector<T> d_A = h_A;
    auto Aptr = thrust::raw_pointer_cast(d_A.data());

    dim3 block(32*4);
    copy_global_shm_register<T, G2SCopy, M, N><<<1, block>>>(Aptr);
}
