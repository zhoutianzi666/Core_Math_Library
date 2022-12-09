#clang-format -style=google -i ./gemm.cu 

nvcc *.cu  -o a.out -arch sm_75 -I /zhoukangkang/2022-04-28inference_try/cutlass/include -lcublas && ./a.out && rm -rf a.out 
nvcc *.cu  -o a.out -arch sm_75 -I /zhoukangkang/2022-05-10Paddle/cutlass/include -lcublas && ./a.out && rm -rf a.out 

