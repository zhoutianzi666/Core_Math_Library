
import paddle




def haha(a,b):
    c = paddle.matmul(a,b)
    c = c + a
    return c



a = paddle.randn([5120,5120], dtype="float16")
b = paddle.randn([5120,5120], dtype="float16")





new0 = paddle.device.Stream()
new1 = paddle.device.Stream()

old = paddle.device.current_stream()
e = paddle.device.Event()
e.record(old)
new0.wait_event(e)
new1.wait_event(e)
paddle.device.set_stream(new0)

for i in range(100):
    out0 = haha(a,b)

e0 = paddle.device.Event()
e0.record(new0)
paddle.device.set_stream(new1)

print("sdcdsv")

for i in range(100):
    out1 = haha(a,b)

e1 = paddle.device.Event()
e1.record(new1)
old.wait_event(e0)
old.wait_event(e1)
paddle.device.set_stream(old)


paddle.device.synchronize()
print(out0-out1)

