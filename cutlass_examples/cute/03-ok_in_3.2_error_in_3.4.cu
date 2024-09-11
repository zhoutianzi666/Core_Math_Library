#include <cuda.h>
#include <stdlib.h>

#include <cute/tensor.hpp>
//git reset --hard v3.2.1
//git reset --hard v3.4.1

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
int main() {
    using T = cute::half_t;
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    // mma
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 1;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1;
    static constexpr int kMmaPN = 1;
    static constexpr int kMmaPK = 2;
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, MMA{});
    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, MMA{});

    // 这个打印的是src到dst的关系哦！
    //print_latex(s2r_tiled_copy_a);
    //print_latex(s2r_tiled_copy_b);
    
    
    print_latex(MMA{});
}
