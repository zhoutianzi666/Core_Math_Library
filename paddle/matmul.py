# required: gpu
import paddle
paddle.set_device("gpu")
m = 512
n = 512
k = 512
WARMUP =  10
REPEATE =  1000

A = paddle.rand((m, k))
B = paddle.rand((k, n))
for i in range(WARMUP):
    A_out = paddle.matmul(A, B)

paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
import datetime
import time
starttime = datetime.datetime.now()

for i in range(REPEATE):
    A_out = paddle.matmul(A, B)
paddle.device.cuda.synchronize(paddle.CUDAPlace(0))

endtime = datetime.datetime.now()
duringtime = endtime - starttime
ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print (ms)# 单位是毫秒
Gflops = REPEATE * (m * n * k * 2 / 1000000) / ms
print("Gflops: ", Gflops);


