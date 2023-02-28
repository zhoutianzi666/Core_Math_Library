#include <cub/cub.cuh>

template <typename T>
using kvp = cub::KeyValuePair<T, T>;

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2> {
  using type = uint16_t;
};
template <>
struct BytesToType<4> {
  using type = uint32_t;
};
template <>
struct BytesToType<8> {
  using type = uint64_t;
};
template <>
struct BytesToType<16> {
  using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data) {
  using T = typename BytesToType<Bytes>::type;

  const T* in = static_cast<const T*>(local);
  T* out = static_cast<T*>(data);
  *out = *in;
}

template <typename T>
struct mySum {
  __host__ __device__ __forceinline__ kvp<T> operator()(kvp<T> const& a,
                                                        kvp<T> const& b) const {
    return kvp<T>(a.key + b.key, a.value + b.value);
  }
};

/*

template <typename T, typename OP_T, int32_t TPB>
__global__ void LayerNormSmallKernel(int32_t const nHiddenDimension,
                                     T const* input,
                                     T const* gamma,
                                     T const* beta,
                                     T* output,
                                     float const epsilon) {
  int32_t const index = blockIdx.x * nHiddenDimension + threadIdx.x;
  T const denominator = T(1.f) ;/// T(nHiddenDimension);
  OP_T val = 0;
  kvp<OP_T> threadData(0, 0);

  if (threadIdx.x < nHiddenDimension) {
    val = input[index];
    OP_T tmp0 = 0;//input[index] * denominator;
    OP_T tmp1 = tmp0 * static_cast<OP_T>(input[index]);
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp0, tmp1));
  }

  using WarpReduce = cub::WarpReduce<kvp<OP_T>, TPB>;
  __shared__ typename WarpReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  auto const sumKV = WarpReduce(temp).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    printf("%f\n", mu);
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

  if (threadIdx.x < nHiddenDimension) {
    OP_T const g = gamma[threadIdx.x], b = beta[threadIdx.x];
    output[index] = (val - mu) * rsigma * g + b;
  }
}

*/

// template __global__ void LayerNormSmallKernel<float, float, 32>(int32_t const,
//                                                                 float const*,
//                                                                 float const*,
//                                                                 float const*,
//                                                                 float*,
//                                                                 float const);
// template __global__ void LayerNormSmallKernel<__half, float, 32>(int32_t const,
//                                                                  __half const*,
//                                                                  __half const*,
//                                                                  __half const*,
//                                                                  __half*,
//                                                                  float const);

template <typename T, typename OP_T, int32_t TPB, int32_t VPT>
__global__ void LayerNormMediumKernel(int32_t const nHiddenDimension,
                                      T const* input,
                                      T const* gamma,
                                      T const* beta,
                                      T* output,
                                      float const epsilon) {
  int32_t const index = blockIdx.x * nHiddenDimension + threadIdx.x * VPT;
  T localX[VPT], localGamma[VPT], localBeta[VPT];
  OP_T const denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  copy<sizeof(T) * VPT>(&input[index], localX);
#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    OP_T const tmp = static_cast<OP_T>(localX[it]) * denominator;
    threadData = mySum<OP_T>()(
        threadData, kvp<OP_T>(tmp, tmp * static_cast<OP_T>(localX[it])));
  }

  copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
  copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ OP_T mu, rsigma;

  auto const sumKV =
      BlockReduce(temp_storage).Reduce(threadData, mySum<OP_T>());
  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

#pragma unroll
  for (int32_t it = 0; it < VPT; it++) {
    localX[it] = static_cast<OP_T>(localGamma[it]) *
                     (static_cast<OP_T>(localX[it]) - mu) * rsigma +
                 static_cast<OP_T>(localBeta[it]);
  }

  copy<sizeof(T) * VPT>(localX, &output[index]);
}

template __global__ void LayerNormMediumKernel<float, float, 64, 4>(
    int32_t const,
    float const*,
    float const*,
    float const*,
    float*,
    float const);
template __global__ void LayerNormMediumKernel<__half, float, 64, 4>(
    int32_t const,
    __half const*,
    __half const*,
    __half const*,
    __half*,
    float const);

template <typename T, typename OP_T, int32_t TPB>
__global__ void LayerNormLargeKernel(int32_t const nHiddenDimension,
                                     T const* input,
                                     T const* gamma,
                                     T const* beta,
                                     T* output,
                                     float const epsilon) {
  int32_t const offset = blockIdx.x * nHiddenDimension;
  OP_T const denominator = OP_T(1) / OP_T(nHiddenDimension);
  kvp<OP_T> threadData(0, 0);

  for (int32_t i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    int32_t const index = offset + i;
    OP_T val = input[index];
    OP_T const tmp = val * denominator;
    threadData = mySum<OP_T>()(threadData, kvp<OP_T>(tmp, tmp * val));
    output[index] = val;
  }

  using BlockReduce = cub::BlockReduce<kvp<OP_T>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ OP_T mu, rsigma;

  auto const sumKV = BlockReduce(temp).Reduce(threadData, mySum<OP_T>());

  if (threadIdx.x == 0) {
    mu = sumKV.key;
    rsigma = rsqrt(sumKV.value - mu * mu + static_cast<OP_T>(epsilon));
  }
  __syncthreads();

  for (int32_t i = threadIdx.x; i < nHiddenDimension; i += TPB) {
    int32_t const index = offset + i;
    output[index] = (static_cast<OP_T>(output[index]) - mu) * rsigma *
                        static_cast<OP_T>(gamma[i]) +
                    static_cast<OP_T>(beta[i]);
  }
}

template __global__ void LayerNormLargeKernel<float, float, 256>(int32_t const,
                                                                 float const*,
                                                                 float const*,
                                                                 float const*,
                                                                 float*,
                                                                 float const);
template __global__ void LayerNormLargeKernel<__half, float, 256>(int32_t const,
                                                                  __half const*,
                                                                  __half const*,
                                                                  __half const*,
                                                                  __half*,
                                                                  float const);

template <typename T>
int32_t layernorm_gpu(int32_t const gridSize,
                         int32_t const nHiddenDimension,
                         T const* input,
                         T const* gamma,
                         T const* beta,
                         T* output,
                         float const epsilon) {
  constexpr int32_t VPT = 16 / sizeof(T);
  if (nHiddenDimension <= 32) {
    constexpr int32_t TPB = 32;
    // (LayerNormSmallKernel<T, float, TPB>)<<<gridSize, TPB, 0>>>(
    //     nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 320) {
    constexpr int32_t TPB = 320 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 640) {
    constexpr int32_t TPB = 640 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 768) {
    constexpr int32_t TPB = 768 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else if (nHiddenDimension == 1280) {
    constexpr int32_t TPB = 1280 / VPT;
    (LayerNormMediumKernel<T, float, TPB, VPT>)<<<gridSize, TPB, 0>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  } else {
    constexpr int32_t TPB = 256;
    (LayerNormLargeKernel<T, float, TPB>)<<<gridSize, TPB, 0>>>(
        nHiddenDimension, input, gamma, beta, output, epsilon);
  }
  // PLUGIN_CHECK_CUDA(cudaPeekAtLastError());
  return 0;
}

template int layernorm_gpu<half>(int const,
                                    int const,
                                    half const*,
                                    half const*,
                                    half const*,
                                    half*,
                                    float const);
