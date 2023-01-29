
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"

using DATATYPE = int;

template <typename Iterator>
__global__ void copy(
    typename Iterator::Params dst_params,
    typename Iterator::Element *dst_pointer,
    typename Iterator::Params src_params,
    typename Iterator::Element *src_pointer,
    cutlass::Coord<2> extent) {


    Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
    Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

    // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
    // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
    // dimension can be accessed via Iterator::Shape::kStrided
    int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

    typename Iterator::Fragment fragment;

    for(int i = 0; i < fragment.size(); ++i) {
      fragment[i] = 0;
    }

    src_iterator.load(fragment);
    dst_iterator.store(fragment);


    ++src_iterator;
    ++dst_iterator;

    for(; iterations > 1; --iterations) {

      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}


int main1(void) {
  int M = 900;
  int N = 60;

  // Note input is in CPU place
  DATATYPE *input;
  int input_size = M * N;

  cudaError_t status = cudaMallocHost(&input, sizeof(DATATYPE) * input_size);
  assert(status == cudaSuccess);
  //init(input, input_size);

  // dev_input is from weight
  DATATYPE *dev_input, *dev_out;
  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, input_size * sizeof(DATATYPE));
  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);


  using Shape = cutlass::layout::PitchLinearShape<16, 4>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int;
  int const kThreads = 32;
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;
  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 1, ThreadMap>;
  typename Iterator::Params dst_params({M});
  typename Iterator::Params src_params({M});

  std::cout << Iterator::TileAccessIterator::UnderlyingPredicates::kAccessesPerVector << std::endl;
  std::cout << Iterator::TileAccessIterator::UnderlyingPredicates::kPredicateWordCount << std::endl;

  
  dim3 block(kThreads, 1);
  dim3 grid(1, 1);
  copy<Iterator><<< grid, block >>>(
          dst_params,
          dev_out,
          src_params,
          dev_input,
          cutlass::make_Coord(M, N));




  cudaDeviceReset();
  cudaFreeHost(input);
  cudaFree(dev_input);
  cudaFree(dev_out);
  return 0;
}
