#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/half.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"


using DATATYPE = half;

int const tb_x = 32;
int const tb_y = 32;

template <typename Iter_out, typename Iter_in, typename Iter_smem, typename Iter_smem_read>
__global__ void copy(
    typename Iter_out::Params dst_params,
    typename Iter_out::Element *dst_pointer,
    typename Iter_in::Params src_params,
    typename Iter_in::Element *src_pointer, int strided, int contiguous) {
    
    __shared__ cutlass::half_t aTile[tb_x][tb_y];

    auto tb_coord = cutlass::make_Coord((int)blockIdx.x * tb_x, (int)blockIdx.y * tb_y);
    auto tb_coord1 = cutlass::make_Coord((int)blockIdx.y * tb_y, (int)blockIdx.x * tb_x);
    Iter_smem smem_iterator({(cutlass::half_t*)aTile, tb_y} ,threadIdx.x);
    typename Iter_smem::Fragment fragment_smem;

    Iter_in src_iterator(src_params, src_pointer, cutlass::make_Coord(contiguous, strided), threadIdx.x, tb_coord);

    typename Iter_in::Fragment fragment_in;
    src_iterator.load(fragment_in);

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == -1)
    for (int i = 0; i < fragment_in.size(); i++) {
        printf("%f\n", float(fragment_in[i]));
    }
    
    smem_iterator.store(fragment_in);
    
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == -1)
    {
        for (int i = 0; i < 32; i++)
        {
            for (int j = 0; j < 32; j++)
            {
                printf("%f\n", (float)(aTile[i][j]));
            }
        }
    }

    // 下面这里还需要搞一个迭代器！
    // Iter_in src_iterator1(src_params, (cutlass::half_t*)(&aTile[0][0]), cutlass::make_Coord(contiguous, strided), threadIdx.x);
     
    Iter_smem_read smem_iterator_read({(cutlass::half_t*)aTile, tb_y} ,threadIdx.x);
    typename Iter_smem_read::Fragment fragment_smem_read;

    smem_iterator_read.load(fragment_smem_read);

    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == -1)
    for (int i = 0; i < fragment_smem_read.size(); i++)
    {
        printf("%f\n", float(fragment_smem_read[i]));
    }

    Iter_out dst_iterator(dst_params, dst_pointer, cutlass::make_Coord(contiguous, strided), threadIdx.x, tb_coord1);
    dst_iterator.store(fragment_smem_read);
}

void cutlass_iter(DATATYPE *output, const DATATYPE *input, int batch, int strided, int contiguous){

    using Shape = cutlass::layout::PitchLinearShape<tb_x, tb_y>;
    using Layout = cutlass::layout::RowMajor;
    using Element = cutlass::half_t;
    int const kThreads = 32 * 4;
    using ThreadMap_out = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
    using ThreadMap_in = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
    
    using Shape1 = cutlass::MatrixShape<tb_x, tb_y>;
    using ThreadMap_smem_w_tmp = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
    using arrange = cutlass::layout::PitchLinearShape<32, 1>;
    using SmemThreadMap_w = cutlass::transform::TransposePitchLinearThreadMapSimt<ThreadMap_smem_w_tmp>;
    // using SmemThreadMap_w = cutlass::transform::PitchLinearWarpRakedThreadMap<Shape, kThreads, 
    // cutlass::layout::PitchLinearShape<1, 32>, 1>;
    using SmemIterator_w = cutlass::transform::threadblock::RegularTileIterator<Shape1, Element, cutlass::layout::RowMajor, 1, SmemThreadMap_w>;
    
    using ThreadMap_smem_read = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
    using SmemIterator_r = cutlass::transform::threadblock::RegularTileIterator<Shape1, Element, cutlass::layout::ColumnMajor, 1, ThreadMap_smem_read>;
    
    using Iterator_in = cutlass::transform::threadblock::PredicatedTileIterator<Shape1, Element, Layout, 0, ThreadMap_in>;
    using Iterator_out = cutlass::transform::threadblock::PredicatedTileIterator<Shape1, Element, Layout, 0, ThreadMap_out>;

    typename Iterator_out::Params dst_params(contiguous);
    typename Iterator_in::Params src_params(contiguous);

    dim3 block(kThreads, 1);
    dim3 grid((strided + tb_x - 1) / tb_x, (contiguous + tb_y - 1) / tb_y, 1);

    // Launch copy kernel to perform the copy
    copy<Iterator_out, Iterator_in, SmemIterator_w, SmemIterator_r><<< grid, block >>>(
            dst_params,
            (cutlass::half_t *)output,
            src_params,
            (cutlass::half_t *)input,
            strided, contiguous);

}

