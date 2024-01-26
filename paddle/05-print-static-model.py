
import paddle
import numpy as np

paddle.enable_static()

# startup_prog = paddle.static.default_startup_program()
# main_prog = paddle.static.default_main_program()
# with paddle.static.program_guard(main_prog, startup_prog):
#     image = paddle.static.data(name="img", shape=[64, 784])
#     w = paddle.create_parameter(shape=[784, 200], dtype='float32')
#     b = paddle.create_parameter(shape=[200], dtype='float32')
#     hidden_w = paddle.matmul(x=image, y=w)
#     hidden_b = paddle.add(hidden_w, b)
exe = paddle.static.Executor(paddle.CPUPlace())
#exe.run(startup_prog)

# 确保这个路径下既有*.pdiparams也有*.pdmodel文件
path_prefix = "models/1122/eva/eval_fix_export_bug/main"

[inference_program, feed_target_names, fetch_targets] = (
    paddle.static.load_inference_model(path_prefix, exe))

print(inference_program)


