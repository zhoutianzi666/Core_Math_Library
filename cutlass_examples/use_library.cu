/*

nvcc use_library.cu -o a.out -arch sm_75  -I/zhoukangkang/2022-04-28inference_try/cutlass/tools/library/include -I/zhoukangkang/2022-04-28inference_try/cutlass/include/ -L/zhoukangkang/2022-04-28inference_try/cutlass/tools/library/ -lcutlass
export LD_LIBRARY_PATH=/zhoukangkang/2022-04-28inference_try/cutlass/tools/library/:$LD_LIBRARY_PATH
tile_description
*/

#include "cutlass/library/library.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/manifest.h"
#include <iostream>
using namespace cutlass;
int main(void) {
    library::Manifest const &manifest = library::Singleton::get().manifest;
    std::cout << &manifest << std::endl;

    for (auto const & operation_ptr : manifest) {
        if(library::OperationKind::kConv2d == operation_ptr->description().kind)
        {
            std::cout << operation_ptr->description().name << "哈哈哈哈" << std::endl;
            std::cout << (int)(((library::ConvDescription*)(&(operation_ptr->description())))->conv_kind) << "哈哈哈哈" << std::endl;
            std::cout << operation_ptr->description().tile_description.minimum_compute_capability << std::endl;
            std::cout << operation_ptr->description().tile_description.minimum_compute_capability << std::endl;
            std::cout << operation_ptr->description().tile_description. maximum_compute_capability << std::endl;
        }
    }


}
