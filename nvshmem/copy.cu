#include <stdio.h>
#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)


__global__ void simple_shift_init(int *destination) {
    int mype = nvshmem_my_pe();
    const int num = 1;
    for(int i = 0; i < num; i++)
    destination[i] = mype;
}

__global__ void simple_shift_print(int *destination) {
    int mype = nvshmem_my_pe();
    const int num = 1;
    for(int i = 0; i < num; i++) {
        printf("simple_shift_print ： %d\n", destination[i]);
    }
}

int main (int argc, char *argv[]) {
    int mype_node, msg;
    cudaStream_t stream;
    int rank, nranks;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    printf("mype_node %d\n", mype_node);

    CUDA_CHECK(cudaSetDevice(mype_node));
    CUDA_CHECK(cudaStreamCreate(&stream));
    const int nums = 64 * 7168;
    int *destination = (int *) nvshmem_malloc (sizeof(int) * nums);

    simple_shift_init<<<1, 1, 0, stream>>>(destination);

    nvshmem_quiet();
    nvshmem_fence();
    int tiotal_cards = nvshmem_n_pes();
    printf("%d\n", tiotal_cards);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // warm up
        if (rank == 1) {
            nvshmem_putmem_nbi(destination, destination, sizeof(int) * nums, 0);
            nvshmem_putmem_nbi(destination, destination, sizeof(int) * nums, 0);
            nvshmem_putmem_nbi(destination, destination, sizeof(int) * nums, 0);
        }
        nvshmem_quiet();
        nvshmem_fence();
        nvshmem_barrier_all();

    cudaEventRecord(start);

    if (rank == 1) {
        nvshmem_putmem_nbi(destination, destination, sizeof(int) * nums, 0);
    }
    nvshmem_quiet();
    nvshmem_fence();

    cudaEventRecord(stop);
    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    if (rank == 0 or rank == 1) {
        printf("time: %f ms\n", time_ms);
        printf("bandwidth: %f GB/s\n", (float)(nums) * sizeof(int) / 1024/1024/1024 / time_ms * 1000);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    simple_shift_print<<<1, 1, 0, stream>>>(destination);

    nvshmemx_barrier_all_on_stream(stream);
    CUDA_CHECK(cudaMemcpyAsync(&msg, destination, sizeof(int),
                cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    printf("%d: received message %d\n", nvshmem_my_pe(), msg);

    nvshmem_free(destination);
    nvshmem_finalize();
    MPI_Finalize();
    return 0;
}