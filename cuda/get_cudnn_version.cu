#include <cudnn_v8.h>
#include <iostream>
// nvcc get_cudnn_version.cu -lcudnn
int main(void) {

  int a = cudnnGetVersion();
  std::cout << "cudnn version: " << a << std::endl;  
  return 0;
}
