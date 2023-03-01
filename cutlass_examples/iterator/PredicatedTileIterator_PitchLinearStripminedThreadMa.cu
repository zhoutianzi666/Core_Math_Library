
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


void init(int *a, int size) {
  for (int i = 0; i < size; i++) {
    a[i] = i;
  }
}

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
    //int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;
    int iterations = (extent[0] + Iterator::Shape::kContiguous - 1) / Iterator::Shape::kContiguous;

    typename Iterator::Fragment fragment;
    
    //src_iterator.clear_mask(true);

    for(int i = 0; i < fragment.size(); ++i) {
      fragment[i] = 0;
    }

    src_iterator.load(fragment);
    dst_iterator.store(fragment);

    ++src_iterator;
    ++dst_iterator;

    // if (threadIdx.x == 0 || 1)
    // {
    //   for(int i = 0; i < fragment.size(); ++i) {
    //     printf("%d\n", fragment[i]);
    //   }
    // }

    for(; iterations > 1; --iterations) {

      src_iterator.load(fragment);
      dst_iterator.store(fragment);

      ++src_iterator;
      ++dst_iterator;
    }
}




int main(void) {

  int real_con = 13;
  int real_stride = 4;
  
  int const con = 16;
  int const stride = 4;

  // Note input is in CPU place
  DATATYPE *input;
  int input_size = real_con * real_stride;
  input = (DATATYPE *)malloc(sizeof(DATATYPE) * input_size);
  assert(input);
  init(input, input_size);

  // dev_input is from weight
  DATATYPE *dev_input, *dev_out;
  cudaMalloc((void **)&dev_input, input_size * sizeof(DATATYPE));
  cudaMalloc((void **)&dev_out, input_size * sizeof(DATATYPE));
  cudaMemcpy(dev_input, input, input_size * sizeof(DATATYPE), cudaMemcpyHostToDevice);


  using Shape = cutlass::layout::PitchLinearShape<con, stride>;
  using Layout = cutlass::layout::PitchLinear;
  using Element = int;
  int const kThreads = 32;
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;
  using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 0, ThreadMap>;
  

  // 这里放的是数据的真实layout！！！
  typename Iterator::Params dst_params({real_con});
  typename Iterator::Params src_params({real_con});

  std::cout << Iterator::TileAccessIterator::UnderlyingPredicates::kAccessesPerVector << std::endl;
  std::cout << Iterator::TileAccessIterator::UnderlyingPredicates::kPredicateWordCount << std::endl;

  dim3 block(kThreads, 1);
  dim3 grid(1, 1);
  copy<Iterator><<< grid, block >>>(
          dst_params,
          dev_out,
          src_params,
          dev_input,
          cutlass::make_Coord(real_con, real_stride));

  cudaDeviceReset();
  cudaFreeHost(input);
  cudaFree(dev_input);
  cudaFree(dev_out);
  return 0;
}
