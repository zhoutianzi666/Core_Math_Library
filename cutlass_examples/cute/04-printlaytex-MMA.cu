#include <cuda.h>
#include <stdlib.h>

#include <cute/tensor.hpp>

using namespace cute;
int main() {
    // mma
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 1;
    static constexpr int kMmaEURepeatN = 4;
    static constexpr int kMmaEURepeatK = 1;

    using mma_atom_shape = mma_traits::Shape_MNK;
    static constexpr int kMmaPM = 1;
    static constexpr int kMmaPN = 2;
    static constexpr int kMmaPK = 1;
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    
    auto thr_mma = MMA{}.get_thread_slice(0);
   // print(thr_mma);
   // print(MMA{});
    
   
   print_latex(MMA{});

}
