import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from requests import head
from paddle.incubate.nn.functional import (
    variable_length_memory_efficient_attention,)

from paddle.incubate.nn.memory_efficient_attention import memory_efficient_attention
import paddle
batch = 8
seq_len = 512
heads = 64
kv_head = heads
head_dim = 128
q = paddle.rand((batch, seq_len, heads , head_dim),dtype ="float16")
k = paddle.rand((batch, seq_len, kv_head , head_dim),dtype ="float16")
v = paddle.rand((batch, seq_len, kv_head , head_dim),dtype ="float16")


attn_mask2 = paddle.rand([batch, heads, seq_len, seq_len], 'float16')
#attn_mask2 = paddle.rand([batch, heads, seq_len, seq_len], 'float16')


warm_up_times = 50
repeat_times = 100

for i in range(warm_up_times):
    qkv_out0 = memory_efficient_attention(q, k, v, attn_bias=attn_mask2, scale=float(head_dim**-0.5))
paddle.device.synchronize("gpu")

import datetime
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out0 = memory_efficient_attention(q, k, v, attn_bias=attn_mask2, scale=float(head_dim**-0.5))
paddle.device.synchronize("gpu")
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The memory_efficient_attention end to end time : ", time_ms, "ms")

# 下面就是baseline哦！
import numpy as np

q = q.transpose([0, 2, 1, 3])
k = k.transpose([0, 2, 1, 3])
v = v.transpose([0, 2, 1, 3])

k = k.reshape([batch, kv_head, 1, seq_len, head_dim])
k = paddle.tile(k, [1, 1, heads // kv_head, 1, 1])
k = k.reshape([batch, heads, seq_len, head_dim])

v = v.reshape([batch, kv_head, 1, seq_len, head_dim])
v = paddle.tile(v,  [1, 1, heads // kv_head, 1, 1])
v = v.reshape([batch, heads, seq_len, head_dim])

out = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
out = out / (np.sqrt(head_dim))
out += attn_mask2
out = paddle.nn.functional.softmax(out, -1)
out =  paddle.matmul(out, v)
out =  out.transpose([0, 2, 1, 3])



seq_lens = paddle.to_tensor([[seq_len] * batch]).astype("int32")


for i in range(warm_up_times):
    qkv_out1 = variable_length_memory_efficient_attention(
    q, k, v, seq_lens, seq_lens, mask=attn_mask2, scale=float(head_dim**-0.5),).transpose([0, 2, 1, 3])
paddle.device.cuda.synchronize(0)

import datetime
starttime = datetime.datetime.now()
for i in range(repeat_times):
    qkv_out1 = variable_length_memory_efficient_attention(
    q, k, v, seq_lens, seq_lens, mask=attn_mask2, scale=float(head_dim**-0.5),).transpose([0, 2, 1, 3])
paddle.device.cuda.synchronize(0)
endtime = datetime.datetime.now()
duringtime = endtime - starttime
time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
print("The variable_length_memory_efficient_attention end to end time : ", time_ms, "ms")

print(paddle.max(paddle.abs(qkv_out0 - qkv_out1)))

