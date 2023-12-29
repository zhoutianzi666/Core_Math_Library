import paddle
T=31
S=62

mask1 = paddle.ones(shape=[T, S])
for i in range(T):
    mask1[i, i * 2:i * 2 + 2] = 0


mask2 = paddle.ones(shape=[T, S])
shape = [T,S]
a = paddle.arange(T)
b = a * 2
c = b + 1
index1 = paddle.stack([a,b],axis=-1)
index2 = paddle.stack([a,c],axis=-1)

updates = paddle.ones(shape=[T])
output1 = paddle.scatter_nd(index1, updates, shape)
output2 = paddle.scatter_nd(index2, updates, shape)

print((mask2 - output1 - output2-mask1).sum())


