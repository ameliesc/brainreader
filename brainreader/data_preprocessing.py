import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from brainreader.pretrained_networks import get_vgg_net
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
import h5py
from scipy.misc import imresize


def get_data(subject = 'S1',  response = 0, data = "train", roi = 1):
    """input: subject = 'S1', stimuli = 1, response = 0, data = "train", roi = 1
    returns: stimulus and response if specified Returns the training or testing
    data set. Default returns training. For validation set type data= "test".
    """
    stimuli = loadmat('Stimuli.mat')
    if data == "train":
        stim_train = stimuli['stimTrn'] #size 1750 x 128 x 128
        if response is 1:
            response = h5py.File('EstimatedResponses.mat')
            roi_subject = response['roi%s' % (subject)]
            response_train = response['dataTrn%s' % (subject)]
            indexes = np.where(roi_subject[0] == roi)[0]
            response_train = np.take(response_train, indexes, axis=1)
            response_train = response_train[:,~np.isnan(response_train).any(0)] #remove Nan 
            return response_train
        else:
            return stim_train
    elif data == "test":
        stim_val = stimuli['stimVal']
        if response is 1:
            response = h5py.File('EstimatedResponses.mat')
            roi_subject = response['roi%s' % (subject)]
            response_val = response['dataVal%s' % (subject)]
            indexes = np.where(roi_subject[0] == roi)[0]
            response_val = np.take(response_val, indexes, axis=1)
            response_val = response_val[:,~np.isnan(response_val).any(0)] #remove Nan 
            return response_val
        else:
            return stim_val
    else:
        raise NameError("%s is not a valid data set" % (data))
