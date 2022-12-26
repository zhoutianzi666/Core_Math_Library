
using DATATYPE = float;

using ACCU_DATATYPE = float;

// 每个cuda thread计算cuda_M * cuda_N 个结果！
#define cuda_M 4
#define cuda_N 4
#define cuda_K 1

#define thread_x_num 16
#define thread_y_num 16

__global__ void kernel_naive1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m,
                              int n, int k) {

  int idx_in_grid = threadIdx.x + blockIdx.x * blockDim.x;
  int idy_in_grid = threadIdx.y + blockIdx.y * blockDim.y;

  // global_row0 和 global_col0是此cuda thread所要处理的4*4的输出的左上角的那个！
  const int global_row0 = idy_in_grid * cuda_M;
  const int global_col0 = idx_in_grid * cuda_N;

  if (global_row0 >= m || global_col0 >= n) return;

  ACCU_DATATYPE sum[cuda_M][cuda_N] = {0.};
  DATATYPE reg_a[cuda_M][cuda_K];
  DATATYPE reg_b[cuda_K][cuda_N];

  for (int i = 0; i < k; i+= cuda_K) {

    for (int ii = global_row0; ii < global_row0 + cuda_M; ii++) {
      for (int jj = i; jj < i + cuda_K; jj++) {
        reg_a[ii - global_row0][jj - i] = a[ii * k + jj];
      }
    }


    for (int ii = global_col0; ii < global_col0 + cuda_N; ii++) {
      for (int jj = i; jj < i + cuda_K; jj++) {
        reg_b[jj - i][ii - global_col0] = b[jj * n + ii];
      }
    }

    for (int reg_i = 0; reg_i < cuda_M; reg_i++) {
      for (int reg_j = 0; reg_j < cuda_N; reg_j++) {
        for (int reg_k = 0; reg_k < cuda_K; reg_k++) {
        sum[reg_i][reg_j] += reg_a[reg_i][reg_k] * reg_b[reg_k][reg_j];
        }
      }
    }
  
  }

  for (int reg_i = 0; reg_i < cuda_M; reg_i++) {
    for (int reg_j = 0; reg_j < cuda_N; reg_j++) {
      c[(global_row0 + reg_i) * n + global_col0 + reg_j] = sum[reg_i][reg_j];
    }
  }
}

void matmul_gpu_naive(DATATYPE *dev_a, DATATYPE *dev_b, DATATYPE *dev_c, int m,
                      int n, int k) {
  uint3 grid = {(m + thread_x_num * cuda_M - 1)/ thread_x_num * cuda_M, 
                (n + thread_y_num * cuda_N - 1)/ thread_y_num * cuda_N,
                  1};
  uint3 block = {thread_x_num, thread_y_num, 1};
  kernel_naive1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
}

