
#include <cub/cub.cuh>

static inline int32_t divUp(int32_t m, int32_t n) { return (m + n - 1) / n; }

static inline __device__ __host__ float sigmoid(float x) {
  return 1.F / (1.F + expf(-x));
}

struct GroupSums {
  // Is it the 1st element of the group?
  int32_t flag;
  float sum;
  float sumSq;
};

struct GroupSumsOp {
  inline __device__ GroupSums operator()(GroupSums const& a,
                                         GroupSums const& b) {
    GroupSums dst;
    dst.sum = b.flag ? b.sum : (a.sum + b.sum);
    dst.sumSq = b.flag ? b.sumSq : (a.sumSq + b.sumSq);
    dst.flag = a.flag + b.flag;
    return dst;
  }
};

struct GroupNormNHWCParams {
  // The output buffer. Layout NHWC.
 float* dst;
  // The output buffer. Layout NHWC.
 // __half* eleOut;
  // The input buffer. Layout NHWC.
  float const* srcX;
  // The input buffer. Layout NHWC.
  // __half const* srcY;
  // The gamma scaling factor.
  void const* gamma;
  // The beta term to add in GN.
  void const* beta;
  // The temporary buffer to do the global parallel reduction. Size:
  // BLOCKS_PER_BATCH x C x 2.
  float* redBuffer;

  int32_t n;
  int32_t h, w;
  int32_t c;
  int32_t groups;
  bool withSwish = false;

  // Precomputed values and parameters to control the execution of the kernels.

  // The number of activations per instance (h * w) and the number of
  // activations per block.
  int32_t hw, hwPerBlock;
  // The number of channels per group and blocks per activation in the C
  // dimension.
  int32_t cPerBlock, cPerGroup;

  // The precomputed stride between instances.
  int32_t hwc;
  // The inverse of hwc in floats (to compute mean/var).
  float invHWC;
  // The precomputed number of groups per block.
  int32_t groupsPerBlock;
  // epsilon, Constant for numerical stability
  float eps;
};


template <int tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCSumKernel(const GroupNormNHWCParams params) {
  // The object in charge of doing the sums for the different blocks.
  typedef cub::BlockScan<GroupSums, tTHREADS_PER_BLOCK> BlockScan;

  // Allocate shared memory for BlockScan.
  __shared__ typename BlockScan::TempStorage tempStorage;
  // Allocate shared memory for the groups. We could reduce the amount of shared
  // memory reserved.
  __shared__ float2 smem[tTHREADS_PER_BLOCK];

  // The instance in the batch.
  int32_t ni = blockIdx.z;

  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // The sums.
  float sum = 0.F;
  float sumSq = 0.F;

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The offset.
    int64_t offset = static_cast<int64_t>(ni) * params.hwc +
                     + ci / 32 * params.hw * 32 + 
                     static_cast<int64_t>(hwi) * 32 + ci % 32;

    // Fetch two channels per thread.
    __half2 h2(0, 0);
    if (ci < params.c) {
      const half* tmp_ptr=(const half*)(params.srcX);
      h2 = *reinterpret_cast<__half2 const *>(&tmp_ptr[offset]);
    }

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    // Update the sum.
    sum += f2.x + f2.y;
    // Update the sum of squares.
    sumSq += f2.x * f2.x + f2.y * f2.y;
  }

  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = threadIdx.x * 2 / params.cPerGroup;
  int32_t cj = threadIdx.x * 2 - params.cPerGroup * gi;

  // The data for the summations.
  GroupSums inp{cj == 0 ? 1 : 0, sum, sumSq};

  // Do the segmented scan.
  GroupSums out;
  BlockScan(tempStorage).InclusiveScan(inp, out, GroupSumsOp());

  // Store the results for the groups in shared memory (to produce coalesced
  // stores later).
  // 2 channels per thread
  if (cj == params.cPerGroup - 2) {
    smem[gi] = make_float2(out.sum, out.sumSq);
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The global group index.
  int32_t gj = blockIdx.x * params.groupsPerBlock + threadIdx.x;

  // Threads that have nothing left to do, exit.
  if (threadIdx.x >= params.groupsPerBlock || gj >= params.groups) {
    return;
  }

  // The first threads (those storing to global memory, load the values).
  float2 sums = smem[threadIdx.x];

  // Store to global memory.
  atomicAdd(&params.redBuffer[(2 * ni + 0) * params.groups + gj], sums.x);
  atomicAdd(&params.redBuffer[(2 * ni + 1) * params.groups + gj], sums.y);
}

void groupNormNCHW32Sum(const GroupNormNHWCParams &params) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCSumKernel<160><<<grid, 160, 0>>>(params);
      break;
    case 480:
      groupNormNHWCSumKernel<256><<<grid, 256, 0>>>(params);
      break;
    case 256:
      groupNormNHWCSumKernel<128><<<grid, 128, 0>>>(params);
      break;
    case 128:
      groupNormNHWCSumKernel<64><<<grid, 64, 0>>>(params);
      break;
  }
}


template <int tTHREADS_PER_BLOCK>
__global__ void groupNormNHWCScaleKernel(const GroupNormNHWCParams params) {
  // The instance in the batch.
  int32_t ni = blockIdx.z;
  // The channel loaded by that thread (2 channels per thread for F16x2).
  int32_t ci = blockIdx.x * params.cPerBlock + threadIdx.x * 2;
  // The group that thread works on and the channel in the group (modulus).
  int32_t gi = ci / params.cPerGroup;

  // Load the sum and sum of squares for the group.
  float sum = 0.F, sumSq = 0.F;
  if (gi < params.groups) {
    sum = params.redBuffer[(2 * ni + 0) * params.groups + gi];
    sumSq = params.redBuffer[(2 * ni + 1) * params.groups + gi];
  }

  // Load gamma/beta.
  float2 gammaF2, betaF2;
  if (ci < params.c && 0) {
    gammaF2 = __half22float2(*reinterpret_cast<half2 const *>(
        reinterpret_cast<half const *>(params.gamma) + ci));
    betaF2 = __half22float2(*reinterpret_cast<half2 const *>(
        reinterpret_cast<half const *>(params.beta) + ci));
  }

  // Compute the mean.
  float mean = sum * params.invHWC;
  // Compute the variance.
  float var = sumSq * params.invHWC - (mean * mean);
  // Compute the inverse of the stddev.
  float invStdDev = rsqrtf(var + params.eps);

  // The first activation loaded by that block.
  int32_t hwBegin = blockIdx.y * params.hwPerBlock;
  // The last activation loaded by that block.
  int32_t hwEnd = min(hwBegin + params.hwPerBlock, params.hw);

  // Iterate over the activations to compute the sums.
  for (int32_t hwi = hwBegin; hwi < hwEnd; ++hwi) {
    // The src/dst offset.
    int64_t offset = static_cast<int64_t>(ni) * params.hwc +
                     + ci / 32 * params.hw * 32 + 
                     static_cast<int64_t>(hwi) * 32 + ci % 32;

    // Fetch two channels per thread.
    __half2 h2(0, 0);
    if (ci < params.c) {
      const half* tmp_ptr=(const half*)(params.srcX);
      h2 = *reinterpret_cast<__half2 const *>(&tmp_ptr[offset]);
    }

    // Extract the two half values.
    float2 f2 = __half22float2(h2);

    // Normalize the channels.
    f2.x = (f2.x - mean) * invStdDev;
    f2.y = (f2.y - mean) * invStdDev;

    // Scale by gamma and add beta.
    // f2.x = gammaF2.x * f2.x + betaF2.x;
    // f2.y = gammaF2.y * f2.y + betaF2.y;

    // Apply Swish if needed.
    if (params.withSwish) {
      f2.x = f2.x * sigmoid(f2.x);
      f2.y = f2.y * sigmoid(f2.y);
    }

    // Store the scaled values.
    if (ci < params.c) {
      *reinterpret_cast<__half2 *>(((float*)(params.dst) + offset / 2)) = __float22half2_rn(f2);
    }
  }
}

void groupNormNCHW32Scale(const GroupNormNHWCParams &params) {
  dim3 grid;

  // The number of blocks to compute all the channels.
  grid.x = params.c / params.cPerBlock;
  // The number of blocks to compute all the activations in a given instance.
  grid.y = divUp(params.hw, params.hwPerBlock);
  // The number of instances.
  grid.z = params.n;

  switch (params.cPerBlock) {
    case 320:
      groupNormNHWCScaleKernel<160><<<grid, 160, 0>>>(params);
      break;
    case 480:
      groupNormNHWCScaleKernel<256><<<grid, 256, 0>>>(params);
      break;
    case 256:
      groupNormNHWCScaleKernel<128><<<grid, 128, 0>>>(params);
      break;
    case 128:
      groupNormNHWCScaleKernel<64><<<grid, 64, 0>>>(params);
      break;
    default:
    assert(0);
  }
}


void groupnorm_gpu(half *output, const half *input, int n, int c, int h,
    int w, int groups) {

        GroupNormNHWCParams params;
        params.srcX = (float*)input;
        cudaMalloc((void **)&(params.redBuffer), sizeof(float) * n * groups * 2);
        cudaMemset(params.redBuffer,
          0,
          2 * sizeof(float) *  n * groups);
        params.cPerBlock = 320;
        params.n = n;
        params.c = c;
        params.h = h;
        params.w = w;
        params.hw = h * w;
        params.hwc = h*w*c;
        params.hwPerBlock = h * w;
        params.groupsPerBlock = 32;
        params.groups = groups;
        params.cPerGroup = c / groups;
        params.invHWC = 1.f / (h * w * params.cPerGroup);
        params.eps = 0.00005f;
        params.dst = (float*)output;
        groupNormNCHW32Sum(params);
        groupNormNCHW32Scale(params);
        cudaFree(params.redBuffer);
}


