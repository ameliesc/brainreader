import numpy as np
import theano.tensor as tt


@symbolic
def maxpool_positions(relu_im, stride = (2, 2), region = (2, 2)):
    pooled_im = max_pool_2d(relu_im, ds = region, st = stride)
    positions = tt.grad(pooled_im.sum(), relu_im)
    return positions

def return_switches(relu_im, stride = (2,2), region = (2,2)):
    f = maxpool_positions.compile()
    return f(relu_im, stride, region)

