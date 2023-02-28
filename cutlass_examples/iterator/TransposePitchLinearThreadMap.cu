
// #include <stdio.h>
// #include <assert.h>
// #include <chrono>
// #include <ctime>
// #include <iostream>
// #include <ratio>


// #include "cutlass/transform/pitch_linear_thread_map.h"

// int main(void) {

//   using Shape = cutlass::layout::PitchLinearShape<32, 128>;
//   int const kThreads = 128;

//   using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<Shape, kThreads, 
//    cutlass::layout::PitchLinearShape<4, 8>, 
//    8>;

//   using arrange = cutlass::layout::PitchLinearShape<4, 8>;
//   using ThreadMap1 = cutlass::transform::TransposePitchLinearThreadMap<ThreadMap, arrange>;
//   ThreadMap1 a;

//   for (int i = 0; i < 32 * 4; i++)
//   {
//     auto coord = a.initial_offset(i);
//     std::cout <<coord.contiguous() << " " <<coord.strided() << std::endl;
//   } 

//   return 0;
// }
