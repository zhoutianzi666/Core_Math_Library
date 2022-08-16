#clang-format -style=google -i ./gemm.cu 
#nvcc gemm.cu -o a.out -arch sm_75 -lcublas && ./a.out && rm -rf a.out

#nvcc megengine.cu -o a.out -arch sm_75 -lcublas && ./a.out && rm -rf a.out
#nvcc wmma.cu -o a.out -arch sm_75 -lcublas && ./a.out && rm -rf a.out
# nvcc test.cu utility.cu gemm.cu -o a.out -arch sm_75 -lcublas -I/zhoukangkang/2022-05-10Paddle/cutlass/include/ && ./a.out && rm -rf a.out
nvcc test.cu utility.cu naive_gemm.cu -o a.out -arch sm_75 -lcublas -I/zhoukangkang/2022-05-10Paddle/cutlass/include/ && ./a.out && rm -rf a.out
