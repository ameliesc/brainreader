-from scipy.io import loadmat
import numpy as np


def print_network_layers(networkfile):
    """
    output the network structure of any of the vgg
    """
    
    vgg_s = loadmat('networkfile')
    for i in range(0,vgg_s['layers'].shape[1]):
        print str(vgg_s['layers'][0,i][0, 0][0][0]) + ' '
    