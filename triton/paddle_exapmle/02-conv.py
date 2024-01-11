
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import paddle
import triton
import triton.language as tl

paddle.seed(123)

@triton.jit
def conv_kernel(
    activation_ptr,  
    weight_ptr,
    output_ptr,
    batch, ic, ih, iw,
    oh, ow, oc,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(oh * ow * batch, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(oc, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    offs_batch = offs_m  // (oh * ow)
    offs_oh = offs_m % (oh * ow) // ow
    offs_ow = offs_m % (oh * ow) % ow

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    weight_ptrs = weight_ptr + offs_k[:,None] + offs_n[None,:] * KH * KW * ic
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for kh in range(0, KH):
        for kw in range(0, KW):
            offs_ih = offs_oh * STRIDE_H + kh - PAD_H
            # offs_ih = tl.where(offs_ih < 0, 0, offs_ih)
            # offs_ih = tl.where(offs_ih < ih, offs_ih, ih - 1)
            
            offs_iw = offs_ow * STRIDE_W + kw - PAD_W
            # offs_iw = tl.where(offs_iw < 0, 0, offs_iw)
            # offs_iw = tl.where(offs_iw < iw, offs_iw, iw - 1)
            mask=offs_ih[:, None] < ih and offs_ih[:, None] >= 0
            mask = mask and offs_iw[:, None] < iw
            mask = mask and offs_iw[:, None] >= 0
            
            activation_ptrs = activation_ptr + offs_batch[:,None] * ih * iw * ic + offs_ih[:,None] * iw * ic + offs_iw[:,None] * ic + offs_k[None, :]
            for k in range(0, tl.cdiv(ic, BLOCK_SIZE_K)):
                
                activation = tl.load(activation_ptrs, mask = mask, other=0.0) 
                weight = tl.load(weight_ptrs)
                
                accumulator += tl.dot(activation, weight)
                
                weight_ptrs += BLOCK_SIZE_K
                activation_ptrs += BLOCK_SIZE_K
    
    output_ptrs = output_ptr + offs_m[:, None] * oc + offs_n[None, :]
    tl.store(output_ptrs, accumulator)

def conv(activation_tensor, weight_tensor):
    batch, ih, iw, ic = activation_tensor.shape
    oc = weight_tensor.shape[0]
    stride_h = 2
    stride_w = 2
    pad_h0 = 1
    pad_h1 = 1
    pad_w0 = 1
    pad_w1 = 1
    dilation_h = 1
    dilation_w = 1
    kh = 3
    kw = 3
    oh = (ih + pad_h0 + pad_h1 - dilation_h * (kh - 1) - 1) // stride_h + 1
    ow = (iw + pad_w0 + pad_w1 - dilation_w * (kw - 1) - 1) // stride_w + 1
    output = paddle.zeros([batch, oh, ow, oc])
    M = batch * oh * ow
    N = oc
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    for i in range(100):
        conv_kernel[grid](
            activation_tensor, weight_tensor, output,
            batch, ic, ih, iw,
            oh, ow, oc,
            # 下面是超参
            KH = kh, KW = kw,
            STRIDE_H = stride_h, STRIDE_W = stride_w,
            PAD_H = pad_h0, PAD_W = pad_w0,
            BLOCK_SIZE_M = 128,
            BLOCK_SIZE_N = 128,
            BLOCK_SIZE_K = 16,
        )
    return output


batch = 8
ic = 128
ih = 64
iw = 64
oc = 512
activation_size = (batch, ih, iw, ic)
weight_size = (oc, 3, 3, ic)

x = paddle.rand(activation_size)
y = paddle.rand(weight_size)-0.5
output_triton = conv(x, y)

#print(output_triton)

import paddle
import paddle.nn as nn

paddle.disable_static()
weight_attr = paddle.ParamAttr(name="weight",
                               initializer = paddle.nn.initializer.Assign(y.transpose([0,3,1,2]).numpy()),
                               learning_rate=0.5,
                               regularizer=paddle.regularizer.L2Decay(1.0),
                               trainable=False)
conv = nn.Conv2D(ic, oc, (3, 3), stride = (2,2), padding=1, weight_attr=weight_attr, data_format='NCHW', padding_mode='zeros')

for i in range(100):
    y_var = conv(x.transpose([0, 3, 1, 2]))

print(paddle.max(y_var.transpose([0, 2, 3, 1]) - output_triton))

