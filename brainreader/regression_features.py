import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from brainreader.pretrained_networks import get_vgg_net
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
import h5py
from scipy.misc import imresize
from data_preprocessing import get_data

__author__ = 'amelie'


def setup_net():
    net = get_vgg_net(up_to_layer='fc8')  # See function get_vgg_net for the layers that you can 
    func = net.compile()
    return func

def get_feat(imput_im,func):
    feat = func(imput_im)
    return feat
