
#include <stdio.h>
#include <iostream>

#include "cutlass/transform/pitch_linear_thread_map.h"





int main(void) {
  using Shape = cutlass::layout::PitchLinearShape<16, 8>;
  int const kThreads = 32 * 2;
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
  ThreadMap a;


  ThreadMap::Iterations b;
  std::cout << b.kContiguous << " " << b.kStrided << std::endl;

  for (int i = 0; i < 32 * 2; i++)
  {
    auto coord = a.initial_offset(i);
    std::cout <<coord.contiguous() << " " <<coord.strided() << std::endl;
  } 
  return 0;
}
