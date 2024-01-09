import paddle
from paddle.jit import to_static
paddle.enable_static()

def func(x):
    x = paddle.nn.ReLU()(x)
    x = paddle.static.Print(x)
    return x

x = paddle.ones([1, 2], dtype='float32')
x_v = func(x)
print(x_v) # [[2. 2.]]



# convert to static graph with specific input description
model = paddle.jit.to_static(
    func,
    input_spec=[
        paddle.static.InputSpec(
            shape=[None, 3, None, None], dtype="float32"),  # images
    ])

 # save to static model
save_path = "./checkpoints/infer"
paddle.jit.save(model, save_path)
print(f"static model has been to {save_path}")
