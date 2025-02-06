# python -m paddle.distributed.launch --gpus "0,1,2,3" /root/paddlejob/workspace/env_run/output/A.py
import paddle
import paddle.distributed as dist

dist.init_parallel_env()

import datetime
import time
data = paddle.zeros([1280,2048],dtype="float16")

for i in range(1):

    paddle.device.synchronize()
    starttime = datetime.datetime.now()
    repeate = 1
    for j in range(repeate):
        if dist.get_rank() == 0:
            dist.send(data, dst=1)
        elif dist.get_rank() == 1:
            dist.recv(data, src=0)
    
    if dist.get_rank() == 0:
        dist.send(data, dst=2)
    elif dist.get_rank() == 2:
        dist.recv(data, src=0)
    
    #dist.broadcast(data, src=0)
    #dist.all_reduce(data)

    paddle.device.synchronize()
    endtime = datetime.datetime.now()
    duringtime = endtime - starttime
    time_ms = duringtime.seconds * 1000 + duringtime.microseconds / 1000.0
    daikuan = data.numel() * 2 / time_ms * 1000 / 1e9 * repeate
    print("send receive的带宽", daikuan.item())
    #print("The whoel end to end time : ", time_ms, "ms")


