
import paddle


def exmaple_function(a,b):
    for i in range(100):
        c = paddle.matmul(a,b)
        a1 = c * 0.02 - 0.01
        b1 = c * 0.01
        a = paddle.clip(a1, -1,1)
        b = paddle.clip(b1, -1,1)

    return c

a = paddle.randn([5120,5120], dtype="bfloat16")
b = paddle.randn([5120,5120], dtype="bfloat16")


new = paddle.device.Stream()
old = paddle.device.current_stream()
new.wait_stream(old)

out0 = exmaple_function(a,b)

paddle.device.set_stream(new)

out1 = exmaple_function(a,b)

old.wait_stream(new)
paddle.device.set_stream(old)

paddle.device.synchronize()
print(out0-out1)


