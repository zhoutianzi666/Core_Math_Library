# required: gpu
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import paddle
paddle.set_device("gpu")



def gemm_flops(shape):
    m=0
    n=0
    k=0
    if(type(shape) == list):
        m,n,k = shape[0],shape[1],shape[2]
    else:
        m=shape
        n=shape
        k=shape

    WARMUP =  10
    REPEATE =  100
    A = paddle.rand((m, k)).astype("float16")
    B = paddle.rand((k, n)).astype("float16")
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
    #print (ms)# 单位是毫秒
    Tflops = REPEATE * (m * n * k * 2 / 10**12) / (ms/10**3)
    return Tflops
    #print("Tflops: ", Tflops)

for n in [128, 256, 512, 1024, 2048, 4096]:
    print(gemm_flops(n))


# def f(x, y):
#     return paddle.matmul(x, y)

# model = paddle.jit.to_static(
#     f,
#     input_spec=[paddle.static.InputSpec(shape=[1, 3, 1024, 1024], dtype="float32"),
#     paddle.static.InputSpec(shape=[1, 3, 1024, 1024], dtype="float32"),
#     ],
#     )
# save_path = "./checkpoints/infer"
# paddle.jit.save(model, save_path)
# print(f"static model has been to {save_path}")


