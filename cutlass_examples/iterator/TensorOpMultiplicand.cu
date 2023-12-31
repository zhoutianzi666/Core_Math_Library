#include <stdio.h>
#include <iostream>
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

int main(void) {
  
  // 第一个参数是每个元素是8个bit，所以为了128byte的cache line，所以足够放128个个元素哦！
  // 每个cuda thread能一次访问128bit，足够访问到16个元素呢！
  cutlass::layout::TensorOpMultiplicand<8, 32> layout;
  printf("%d\n", layout.kTileShapeContiguous);
  printf("%d\n", layout.kFactor);

  return 0;
}