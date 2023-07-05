# required: gpu
import paddle

m = 512
n = 512
k = 512
A = paddle.rand((m, k))
B = paddle.rand((k, n))
WARMUP =  10
REPEATE =  1000
for i in range(WARMUP):
    A_out = paddle.matmul(A, B)

paddle.device.cuda.synchronize(paddle.CUDAPlace(0))
import datetime
import time
starttime = datetime.datetime.now()

for i in range(REPEATE):
    A_out = paddle.matmul(A, B)

endtime = datetime.datetime.now()
duringtime = endtime - starttime
print (duringtime.seconds * 1000 + duringtime.microseconds / 1000.0)# 单位是毫秒
paddle.device.cuda.synchronize(paddle.CUDAPlace(0))


