



import paddle
import paddle.distributed as dist

# python -m paddle.distributed.launch --gpus "0,1,2,3" A.py

dist.init_parallel_env()


seq = 1024
data = paddle.randn([1024, 2048])

in_split_sizes = [seq//4] * 4
out_split_sizes = [seq//4] * 4
output_data = paddle.empty_like(data)

for i in range(5):
    dist.alltoall_single(output_data, data, in_split_sizes, out_split_sizes)












