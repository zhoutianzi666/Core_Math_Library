#include <stdio.h>

#include <chrono>
#include <ctime>
#include <iostream>
#include <ratio>

#include "cublas_v2.h"
#include "utility.h"

#define WARMUP 10
#define REPEATE 10

using DATATYPE = float;

// 每个 block 含有block_M * block_N 个结果
// 每个block计算（block_M * cuda_M） * （block_N * cuda_N）个结果呢！
#define block_M 16
#define block_N 16

#define block_K 8
// 每个cuda thread计算cuda_M * cuda_N 个结果呢！
#define cuda_M 8
#define cuda_N 8

__global__ void kernel_gpu(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m, int n,
                           int k) {
  // 这是这个cuda thread要得出的结果是cuda_M * cuda_N 哦！
  // 这里假定是这个cuda thread计算的结果挨在一起呢！
  // row和col显然就是他们的左上角的元素的位置了哦，记住他们是全局位置。
  const int row = (threadIdx.x + blockIdx.x * blockDim.x) * cuda_M;
  const int col = (threadIdx.y + blockIdx.y * blockDim.y) * cuda_N;
  // 显然他们是局部的位置了。
  const int row_in_block = (threadIdx.x) * cuda_M;
  const int col_in_block = (threadIdx.y) * cuda_N;
  // 这个block中左上角的元素的坐标
  const int row0_in_block = (blockIdx.x * blockDim.x) * cuda_M;
  const int col0_in_block = (blockIdx.y * blockDim.y) * cuda_N;
  
  __shared__ DATATYPE aTile[block_K][block_M * cuda_M];
  //__shared__ DATATYPE aTile[block_M * cuda_M][block_K];
  __shared__ DATATYPE bTile[block_K][block_N * cuda_N];
  // 记住C不需要shared memory哦！
  DATATYPE cTile[cuda_M][cuda_N] = {0};
  DATATYPE a_reg[cuda_M];
  DATATYPE b_reg[cuda_N];

  for (int i = 0; i < k; i += block_K) {

    // 每个block计算（block_M* cuda_M）* （block_N * cuda_N）这么多数字！，
    // 也就是block中每个cuda thread计算cuda_M * cuda_N个数字
    int thread_id_in_block = threadIdx.x + threadIdx.y * block_M;
    // printf("%d\n", thread_id_in_block);
    // 下面这两个是典型的条带挖掘哦！
    for(int id = thread_id_in_block; id < block_K * block_M * cuda_M; id += block_M * block_N) {
      int aTile_x = id / block_K;
      int aTile_y = id % block_K;
      // 我好奇，这里难道不会有bank 冲突吗？应该是会有的吧。
      aTile[aTile_y][aTile_x] = a[(aTile_x + row0_in_block) * k + aTile_y + i];
      //aTile[aTile_x][aTile_y] = a[(aTile_x + row0_in_block) * k + aTile_y + i];
    }

    for(int id = thread_id_in_block; id < block_K * block_N * cuda_N; id += block_M * block_N) {
      int bTile_x = id / (block_N * cuda_N);
      int bTile_y = id % (block_N * cuda_N);
      bTile[bTile_x][bTile_y] = b[(i + bTile_x) * n + col0_in_block + bTile_y];
    }

    __syncthreads();

    // 上面结束的时候，数据已经在shared memory啦！
    // 
    // 下面肯定就是要计算这两个矩阵的乘积了
    // 也就是计算一个（block_M* cuda_M） * block_K 与 block_K * （block_N* cuda_N） 的矩阵乘积了
    // 下面就是肯定要考虑block_M* cuda_M）和 （block_N* cuda_M） 这个输出矩阵在thread block里面的划分了哦！
    // 这里的划分方式显然是cuda_M * cuda_N个东西紧紧的挤在一起喽！
    // 绝对不可能是每个cuda thread只算1x1个输出，因为这导致每个cuda thread的计算访存比很低的。
    // 如果有tensor core，我们还应该考虑warp level哦！
    for (int j = 0; j < block_K; j++) {
#pragma unroll
      for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
        a_reg[cTile_i] = aTile[j][row_in_block + cTile_i];
        //a_reg[cTile_i] = aTile[row_in_block + cTile_i][j];
      }
#pragma unroll
      for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
        b_reg[cTile_j] = bTile[j][col_in_block + cTile_j];
      }
#pragma unroll
      for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
        for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
          cTile[cTile_i][cTile_j] += a_reg[cTile_i] * b_reg[cTile_j];
        }
      }
    }
    __syncthreads();
  }

  // 从这里才可以看出来输出结果在thread block上的划分方式。
  for (int cTile_i = 0; cTile_i < cuda_M; cTile_i++) {
    for (int cTile_j = 0; cTile_j < cuda_N; cTile_j++) {
      c[(row + cTile_i) * n + col + cTile_j] = cTile[cTile_i][cTile_j];
    }
  }
}

void matmul_gpu(DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m, int n,
                int k) {
  uint3 grid = {m / (block_M * cuda_M), n / (block_N * cuda_N), 1};
  uint3 block = {block_M, block_N, 1};
  kernel_gpu<<<grid, block, (block_M * cuda_M + block_N * cuda_N) *
                                sizeof(float) * block_K>>>(dev_a, dev_b, dev_c,
                                                           m, n, k);
}
