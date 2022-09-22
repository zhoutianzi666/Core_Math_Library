


# 512 * 512 * 512 的测试

| T4机器  | time/ms  |
| ------- | -------- |
| cublas  | 1.317952 |
| cutlass | 2.614464 |
| meginge | 2.573920 |
| matmul_gpu | 3.028128 |






# 1024 * 1024 * 512 的测试

- 这个情况下，cublas肯定不是用的split-K算法
- 让我们看看这个情况下


| T4机器               | time/ms  | max diff |
| -------------------- | -------- | -------- |
| cublas               | 4.492864 | 0.000076 |
| cutlass              | 5.239712 | 0.000198 |
| matmul_gpu_megengine | 5.462400 | 0.000198 |



| GeForce RTX 2080  SUPER | time/ms  | max diff |
| ----------------------- | -------- | -------- |
| cublas                  | 1.391872 | 0.000122 |
| cutlass                 | 1.864192 | 0.000198 |
| matmul_gpu_megengine    | 1.820352 | 0.000198 |
| matmul_gpu              | 3.171776 | 0.000198 |




- 为啥还是cublas牛逼呢！我郁闷



