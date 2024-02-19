import numpy as np
np.random.seed(20160703)
from numba import cuda
@cuda.jit
def f(a, b, c):
    # like threadIdx.x + (blockIdx.x * blockDim.x)
    #tid = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    tid = cuda.grid(1)
    #tid = cuda.threadIdx.x
    size = len(c)
    if tid < size:
        c[tid] = a[tid] + b[tid]

N = 256
a = cuda.to_device(np.random.random(N))
b = cuda.to_device(np.random.random(N))
c = cuda.device_array_like(a)

f.forall(len(a))(a, b, c)

# Enough threads per block for several warps per block
nthreads = 256
nblocks = (N // nthreads) + 1
f[nblocks, nthreads](a, b, c)

print(c.copy_to_host())
