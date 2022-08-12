#clang-format -style=google -i ./gemm.cu 
nvcc gemm.cu -o a.out -arch sm_75 -lcublas && ./a.out && rm -rf a.out

#nvcc megengine.cu -o a.out -arch sm_75 -lcublas && ./a.out && rm -rf a.out
