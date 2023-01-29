
// #include <stdio.h>
// #include <assert.h>
// #include <chrono>
// #include <ctime>
// #include <iostream>
// #include <ratio>


// #include "cutlass/transform/pitch_linear_thread_map.h"

// int main(void) {
//   using Shape = cutlass::layout::PitchLinearShape<16, 4>;
//   int const kThreads = 32;
//   using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads, 1>;


//   using arrange = cutlass::layout::PitchLinearShape<4, 8>;
//   using ThreadMap1 = cutlass::transform::TransposePitchLinearThreadMap<ThreadMap, arrange>;

//   ThreadMap1 a;

//   auto coord = a.initial_offset(16);
//   std::cout <<coord.contiguous() << std::endl;
//   std::cout <<coord.strided() << std::endl;
//   return 0;
// }
