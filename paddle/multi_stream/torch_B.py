import torch



def haha(a,b):
    for i in range(100):
        c = torch.mm(a,b)
        a1 = c * 0.02 - 0.01
        b1 = c * 0.01
        a = torch.clip(a1, -1,1)
        b = torch.clip(b1, -1,1)
    return c

typ = torch.bfloat16
n = 5120
a = torch.randn(n,n).type(typ).cuda()
b = torch.randn(n,n).type(typ).cuda()



new = torch.cuda.Stream()

old = torch.cuda.current_stream()
new.wait_stream(old)


out0 = haha(a,b)

torch.cuda.set_stream(new)

out1 = haha(a,b)

old.wait_stream(new)
torch.cuda.set_stream(old)

torch.cuda.synchronize()
print(out0-out1)


