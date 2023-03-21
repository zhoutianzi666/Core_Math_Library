
using DATATYPE = int8_t;
using BIAS_DATATYPE = float;
using C_DATATYPE = float;

void init(int8_t *a, int size);
void init(float *a, int size);

int32_t diff(const int32_t *c, const int32_t *c_baseline, int n);
int32_t diff(const int8_t *c, const int32_t *c_baseline, int n);
float diff(const float *c, const float *c_baseline, int n);



template <typename T>
void naive_gemm_cpu(const int8_t *a, const int8_t *b, T *c_cpu, int m,
                    int n, int k, const T * bias);

cudaError_t CutlassIgemmNN(int M, int N, int K,
                           DATATYPE const *A, int lda, 
                           DATATYPE const *B, int ldb, 
                           float const *bias,
                           C_DATATYPE *C, int ldc);

