
using DATATYPE = int8_t;
using BIAS_DATATYPE = float;
using C_DATATYPE = int8_t;

void init(int8_t *a, int size);
void init(float *a, int size);

int32_t diff(const int32_t *c, const int32_t *c_baseline, int n);
int32_t diff(const int8_t *c, const int32_t *c_baseline, int n);

void naive_gemm_cpu(const int8_t *a, const int8_t *b, int32_t *c_cpu_int32, int m,
                    int n, int k);

cudaError_t CutlassIgemmNN(int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B, int ldb, 
                           float const *bias,
                           C_DATATYPE *C, int ldc);

