#include <iostream>
#include <sstream>
#include <vector>

// CUTLASS includes
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear_2dthreadtile.h"


using DATATYPE = half;

int const tb_x = 32;
int const tb_y = 32;

/// Define PredicatedTileIterators to load and store a M-by-K tile, in column major layout.

template <typename Iter_in, typename Iter_out, typename Iter_smem>
__global__ void copy(
    typename Iter_out::Params dst_params,
    typename Iter_out::Element *dst_pointer,
    typename Iter_in::Params src_params,
    typename Iter_in::Element *src_pointer, int strided, int contiguous) {
    
    __shared__ half aTile[tb_x][tb_y];

    auto tb_coord = cutlass::make_Coord((int)blockIdx.x * tb_x, (int)blockIdx.y * tb_y);
    // typename Iter_smem::Params smem_params(contiguous);
    // Iter_out smem_iterator(dst_params, dst_pointer, cutlass::make_Coord(contiguous, strided), threadIdx.x, tb_coord);

    Iter_out dst_iterator(dst_params, dst_pointer, cutlass::make_Coord(contiguous, strided), threadIdx.x, tb_coord);
    Iter_in src_iterator(src_params, src_pointer, cutlass::make_Coord(contiguous, strided), threadIdx.x, tb_coord);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    int iterations = 4;

    typename Iter_in::Fragment fragment_in;
    // printf("%d\n", fragment_in.size());
    src_iterator.load(fragment_in);
    dst_iterator.store(fragment_in);
}

void cutlass_iter(DATATYPE *output, const DATATYPE *input, int batch, int strided, int contiguous){

    using Shape = cutlass::layout::PitchLinearShape<tb_x, tb_y>;
    using Layout = cutlass::layout::PitchLinear;
    using Element = cutlass::half_t;
    int const kThreads = 32 * 4;
    using ThreadMap_in = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 4>;
    using ThreadMap_out = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 4>;
    
    using ThreadMap_smem_in = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
    using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<ThreadMap_smem_in>;
    using SmemIteratorA = cutlass::transform::threadblock::RegularTileIterator<
    Shape, Element, cutlass::layout::ColumnMajor, 1, SmemThreadMapA>;



    using Iterator_in = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 0, ThreadMap_in>;
    using Iterator_out = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 1, ThreadMap_out>;



    typename Iterator_in::Params dst_params(contiguous);
    typename Iterator_out::Params src_params(contiguous);

    dim3 block(kThreads, 1);
    dim3 grid((strided + tb_x - 1) / tb_x, (contiguous + tb_y - 1) / tb_y, 1);

    // Launch copy kernel to perform the copy
    copy<Iterator_out, Iterator_in, SmemIteratorA><<< grid, block >>>(
            dst_params,
            (cutlass::half_t *)output,
            src_params,
            (cutlass::half_t *)input,
            strided, contiguous);

}

