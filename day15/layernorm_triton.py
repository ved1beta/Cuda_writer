import triton 
import torch 
import triton.language as tl 

DEVICE = "cuda"

@triton.jit 
def _layer_norm_fwd_kernal(
    input, 
    output, 
    weights, 
    bias, 
    MEAN, 
    Rstd ,
    stride, 
    N, 
    esp, 
    BLOCK_SIZE:tl.constexpr ,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride 

    mean = 0 
    _mean = tl.zeros([BLOCK_SIZE], dtype = tl.float32)

    for off in range( 0 , N , BLOCK_SIZE):
        cols = off * tl.arange(0, BLOCK_SIZE)
        x = tl.load(input + cols , mask = cols < N , other= 0 ).to(tl.float32)
        _mean += a 
    mean = tl.sum(_mean , axis = 0 )/N

    _var = 0 
    for off in range(0, N , BLOCK_SIZE):
        cols = off * tl.arange(0)
        y = tl.load(input +cols, mask = cols < N, other= 0 )
        y = tl.where(cols < N , x - mean , 0.)
        _val = y * y 
    var +=tl.sum( _val , axis = 0 )/N
    stand = 1/ tl.sqrt(var+ esp)

    tl.store(MEAN + row, mean)
    tl.store(Rstd + row, stand)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(weights + cols, mask=mask)
        b = tl.load(bias + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * stand
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)






    
