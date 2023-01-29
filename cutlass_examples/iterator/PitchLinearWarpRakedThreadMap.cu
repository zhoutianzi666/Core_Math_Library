
#include <stdio.h>
#include <assert.h>
#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>


#include "cutlass/transform/pitch_linear_thread_map.h"

int main3(void) {
  using Shape = cutlass::layout::PitchLinearShape<8, 16>;
  int const kThreads = 32 * 1;

  using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<Shape, kThreads, 
   cutlass::layout::PitchLinearShape<4, 8>, 
   1>;
  ThreadMap a;
  
  for (int i = 0; i < 32 * 1; i++)
  {
    auto coord = a.initial_offset(i);
    std::cout <<coord.contiguous() << " " <<coord.strided() << std::endl;
  } 
  return 0;
}
