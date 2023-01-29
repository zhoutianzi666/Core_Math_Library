
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>


#include "cutlass/transform/pitch_linear_thread_map.h"

int main2(void) {
  using Shape = cutlass::layout::PitchLinearShape<16, 4>;
  int const kThreads = 32;
  using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;
  ThreadMap a;
  ThreadMap::Iterations b;

  std::cout << b.kContiguous << std::endl;
  std::cout << b.kStrided << std::endl;




  auto coord = a.initial_offset(16);
  std::cout <<coord.contiguous() << std::endl;
  std::cout <<coord.strided() << std::endl;
  return 0;
}
