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
// 
int main() {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int N1 = 32;
    
    using SmemLayoutAtom = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(make_shape(Int<8>{}, Int<N1>{}),
                    make_stride(Int<N1>{}, Int<1>{})
                    )));
    using SmemLayout = decltype(tile_to_shape(SmemLayoutAtom{},
                                              make_shape(Int<M>{}, Int<N>{})));

    PRINT("SmemLayout", SmemLayout{}.shape());
    PRINT("SmemLayout", SmemLayout{});
}
