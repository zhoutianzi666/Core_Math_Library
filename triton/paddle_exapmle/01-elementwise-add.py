
import paddle
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    #output = tl.math.exp2(x + y)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x, y):
    output = paddle.zeros(x.shape)
    n_elements = 98432
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output



size = 98432
x = paddle.rand((size,))
y = paddle.rand((size,))
output_paddle = x + y
output_triton = add(x, y)
print(output_paddle)
print(output_triton)
