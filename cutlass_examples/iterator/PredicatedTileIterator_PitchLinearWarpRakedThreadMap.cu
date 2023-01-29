
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>


#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/transform/pitch_linear_thread_map.h"


int main(void) {

   
  int const kThreads = 32 * 4;
  using Iterator = 
  cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<128, 32>, 
  cutlass::half_t, 
  cutlass::layout::RowMajor, 
  1, 

  cutlass::transform::PitchLinearWarpRakedThreadMap<cutlass::PitchLinearShape<32, 128>, 
                                                     kThreads, 
                                                     cutlass::PitchLinearShape<4, 8>, 
                                                     8>, 

  8, 
  0>;

  typename Iterator::Fragment fragment;
  
  // 32 * 128 / 128 = 32
  std::cout << fragment.size() << std::endl;
  
  return 0;
}
