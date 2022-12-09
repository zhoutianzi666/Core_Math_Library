using DATATYPE = float;

using ACCU_DATATYPE = float;

// 每个cuda thread计算cuda_M * cuda_N 个结果！
#define cuda_M 4
#define cuda_N 4
#define cuda_K 4

__global__ void kernel_naive1(DATATYPE *a, DATATYPE *b, DATATYPE *c, int m,
                              int n, int k) {

  int id_in_grid = threadIdx.x + blockIdx.x * blockDim.x;
  // n_need_cuda_thread: nn 这个方向上需要的 cuda 线程数目
  int n_need_cuda_thread = (n + cuda_N - 1) / cuda_N;

  // global_row0 和 global_col0是此cuda thread所要处理的4*4的输出的左上角的那个！
  const int global_row0 = id_in_grid / n_need_cuda_thread * cuda_M;
  const int global_col0 = id_in_grid % n_need_cuda_thread * cuda_N;

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
  uint3 grid = {m * n / (16 * cuda_M * cuda_N) + 1, 1, 1};
  uint3 block = {16, 1, 1};
  kernel_naive1<<<grid, block>>>(dev_a, dev_b, dev_c, m, n, k);
}
