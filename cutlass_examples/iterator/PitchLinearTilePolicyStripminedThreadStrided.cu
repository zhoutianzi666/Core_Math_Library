
// #include <stdio.h>
// #include <assert.h>
// #include <chrono>
// #include <ctime>
// #include <iostream>
// #include <ratio>


// #include "cutlass/transform/pitch_linear_thread_map.h"

// int main(void) {
//   using Shape = cutlass::layout::PitchLinearShape<1, 64 * 64>;
//   int const kThreads = 128;

//   using ThreadMap = cutlass::transform::PitchLinearTilePolicyStripminedThreadStrided<Shape, kThreads, 
//    1>;
//   ThreadMap a;
  
//   for (int i = 0; i < 32; i++)
//   {
//     auto coord = a.initial_offset(i);
//     std::cout <<coord.contiguous() << " " <<coord.strided() << std::endl;
//   }
//   return 0;
// }
