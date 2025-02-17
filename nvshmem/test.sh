

# 分别配置nvshmem, mpi, cuda, nccl的路径
export NVSHMEM_HOME=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/nvshmem_src_3.0.6-4/
export MPI_HOME=/usr/mpi/gcc/openmpi-4.1.5a1/
export NCCL_HOME=/root/paddlejob/workspace/env_run/output/zkk/nvshmem/nccl/build/

# 将nvshmem, mpi, nccl的头文件和库文件都添加到路径中
export CPATH=$CPATH:$NVSHMEM_HOME/include:$MPI_HOME/include:$NCCL_HOME/include
export LIBRARY_PATH=$LIBRARY_PATH:$NVSHMEM_HOME/lib:$MPI_HOME/lib:$NCCL_HOME/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NVSHMEM_HOME/lib:$MPI_HOME/lib:$NCCL_HOME/lib

# 配置编译的其他参数
export NVCC_GENCODE="arch=compute_90,code=sm_90"
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_USE_NCCL=1

# 编译执行
#nvcc -rdc=true -ccbin g++ -gencode=$NVCC_GENCODE test.cu -o test.out -lnvshmem_host -lnvshmem_device -lmpi 
mpirun -n 2 ./test.out
# #./test.out
