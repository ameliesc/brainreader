import numpy as np
import theano.tensor as tt
from plato.core import symbolic
from theano.tensor.signal.downsample import max_pool_2d


@symbolic
def maxpool_positions(relu_im, stride = (2, 2), region = (2, 2)):
    pooled_im = max_pool_2d(relu_im, ds = region, st = stride)
    positions = tt.grad(pooled_im.sum(), relu_im)
    return positions

# def return_switches(relu_tensor, stride = (2,2), region = (2,2)):
#     """
#     relu_tensor: (n_samples, n_input_maps, size_y, size_x) 
#     """
    
#     relu_array = relu_tensor.eval()
#     print relu_array.shape
#     switch_array = np.zeros_like(relu_array)
#     f = maxpool_positions.compile()
#     for i in range(0,relu_array.shape[1]):
#         switch = f(relu_array[0][i])
#         switch_array[0][i] = switch
#     return switch_array

