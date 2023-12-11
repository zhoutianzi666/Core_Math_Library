import paddle
from paddle.incubate.nn.functional import masked_multihead_attention

heads = 16
kv_head = heads
head_dim = 16
max_seq_len = 1280
max_batch = 2
cache_kv = paddle.rand((2, max_batch, kv_head, max_seq_len, head_dim),dtype ="float32")
batch = 2
q_len = 1

# 一共要解码100次哦！
decoder_len = 100
Q = paddle.rand((decoder_len, batch, heads, 1, head_dim),dtype ="float32")
K = paddle.rand((decoder_len, batch, heads, 1, head_dim),dtype ="float32")
V = paddle.rand((decoder_len, batch, heads, 1, head_dim),dtype ="float32")

ATTN_MASK = paddle.rand([batch,1,1,decoder_len],'float32')


my_all_out = paddle.empty((0, heads * head_dim),dtype ="float32")
your_all_out = paddle.empty((0, heads * head_dim),dtype ="float32")


for i in range(decoder_len):
    q = Q[i]
    k = K[i]
    v = V[i]
    qkv = paddle.concat([q, k, v], axis=0)
    qkv = qkv.reshape([3, batch, heads, q_len, head_dim])
    qkv_out = qkv.transpose([1, 3, 0, 2, 4]).reshape([batch,3 * heads * head_dim])

    seq_lens = paddle.to_tensor([[i], [i]]).astype("int32")

    attn_mask = ATTN_MASK[:,:,:,:i+1]

    fmha_out = masked_multihead_attention(
        x=qkv_out,
        src_mask=attn_mask,
        cache_kv=cache_kv,
        sequence_lengths=seq_lens)[0]
    my_all_out = paddle.concat([fmha_out,my_all_out],axis=0)




# 下面就是baseline哦！


cache_k = paddle.empty((batch, heads, 0, head_dim),dtype ="float32")
cache_v = paddle.empty((batch, heads, 0, head_dim),dtype ="float32")

for i in range(decoder_len):
    import numpy as np
    
    print("哈哈", i)

    q = Q[i]
    k = K[i]
    v = V[i]

    seq_len = 1
    k = k.reshape([batch, kv_head, 1, seq_len, head_dim])
    k = paddle.tile(k, [1, 1, heads // kv_head, 1, 1])
    k = k.reshape([batch, heads, seq_len, head_dim])
    cache_k = paddle.concat([cache_k, k], axis=-2)
    v = v.reshape([batch, kv_head, 1, seq_len, head_dim])
    v = paddle.tile(v,  [1, 1, heads // kv_head, 1, 1])
    v = v.reshape([batch, heads, seq_len, head_dim])
    cache_v = paddle.concat([cache_v, v], axis=-2)

    out = paddle.matmul(q, cache_k.transpose([0, 1, 3, 2]))
    out = out / (np.sqrt(head_dim))
    out += ATTN_MASK[:,:,:,:i+1]
    out = paddle.nn.functional.softmax(out, -1)
    out =  paddle.matmul(out, cache_v)
    out = out.reshape([batch, -1])
    your_all_out = paddle.concat([out,your_all_out],axis=0)


print(paddle.max(my_all_out - your_all_out))


