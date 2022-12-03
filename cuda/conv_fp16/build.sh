#clang-format -style=google -i ./*.cu 

nvcc *.cu  -o a.out -arch sm_75 -I /zhoukangkang/2022-04-28inference_try/cutlass/include -lcublas -lcudnn -I /zhoukangkang/2022-04-28inference_try/cutlass/tools/util/include/ && ./a.out && rm -rf a.out
