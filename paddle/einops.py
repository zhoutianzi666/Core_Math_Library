

import einops
import paddle









f = 2
h = 2
w = 2
p1 = 3
p2 = 3
p3 = 3
C = 8
x = paddle.randn((1, f*h*w, p1*p2*p3*C), dtype=paddle.float16)

y = einops.rearrange(x, "B (f h w) (p1 p2 p3 C) -> B C (f p1) (h p2) (w p3)", 
                        f=2, 
                        h=2, 
                        w=2, 
                        p1=3, p2=3, p3=3)

x = x.reshape([x.shape[0], f, h, w,p1, p2, p3, C])
x = paddle.transpose(x, [0, 7, 1, 4, 2, 5, 3, 6])
x = x.reshape([1, C, f*p1, h*p2, w*p3])

print(y-x)

