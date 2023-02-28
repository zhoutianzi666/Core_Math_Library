// #include <stdio.h>
// #include <iostream>
// #include "cutlass/transform/pitch_linear_thread_map.h"

// int main(void) {
//   using Shape = cutlass::layout::PitchLinearShape<32, 128>;
//   int const kThreads = 128;

//   using ThreadMap = cutlass::transform::PitchLinearWarpRakedThreadMap<Shape, kThreads, 
//    cutlass::layout::PitchLinearShape<1, 32>, 
//    2>;
//   ThreadMap a;
  
//   ThreadMap::Iterations b;
//   std::cout << b.kContiguous << " " <<  b.kStrided << std::endl;
  
//   for (int i = 0; i < 32 * 4; i++)
//   {
//     auto coord = a.initial_offset(i);
//     std::cout <<coord.contiguous() << " " <<coord.strided() << std::endl;
//   } 
//   return 0;
// }
