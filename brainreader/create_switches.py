import numpy as np
import theano.tensor as tt


def create_switches(relu_out, pool_out, stride, region):
    """
    Input:
    - relu_out: convnet relu-later output preceeding pool-layer
    - pool_out: convnet pool-layer outpus
    - stride: pooling stride
    - region: pooling region
    Output:
    Numpy matrix with ones indicating the maximum value of pooling region,
    same dimensions as relu_out
    """
    pool = pool_out[0][0]
    relu = relu_out[0][0]
    rowp, columnp = pool.shape[-2:]
    rowr, columnr = relu.shape[-2:]
    k = 0
    region_end_k = range(region[0])[-1]
    region_end_l = range(region[1])[-1]
    switch_matrix = np.zeros_like(relu)
    print type(switch_matrix[0][1])
    i = 0
    while i < rowp: 
        j = 0
        region_beg_k = k
        region_end_k = region_beg_k + range(region[0])[-1]
        while j < columnp:
            max_val = pool[i][j]
            l = 0
            while k < region_end_k:
                region_beg_l = l
                region_end_l = region_beg_l + range(region[1])[-1]
                while l < region_end_l:
                    ## need to account for the case that same number occurs twice
                    if relu[k][l] ==  max_val:
                        switch_matrix[k][l] = 1.0
                        print switch_matrix[k][l], k, l
                    l += 1
                l = region_beg_l + stride[1]
                k += 1
            k = region_beg_k + stride[0]
            j +=1
        i +=1
    return switch_matrix
