import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from pretrained_networks import get_vgg_net
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
import h5py
from scipy.misc import imresize
from data_preprocessing import get_data

__author__ = 'amelie'


def im2feat(im):
    im = imresize(im, (224, 224))
    np.transpose(np.dstack((im, im, im)))
    return im
# prepcossing of image :
# 1 smooth
# remove the mean


def feat2im(feat): return feat.dimshuffle(0, 2, 3, 1)[0, :, :, :]\
    if isinstance(feat, Variable) else np.rollaxis(feat, 0, 2)[0, :, :, :]


def normalize(arr): return (arr - np.mean(arr)) / np.std(arr)


def setup_net():
    net = get_vgg_net(up_to_layer='fc8')
    # See function get_vgg_net for the layers that you can
    func = net.compile()
    return func


def demo_brainreader():
    stimuli_train, response_train = get_data(response=1)

    func = setup_net()
    imput_im = np.empty([1, 3, 224, 224])
    imput_im[0] = normalize(im2feat(stimuli_train[0]))
    feat = np.squeeze(func(imput_im))
    regr_x = np.empty([1750, feat.shape[0]])
    regr_x[0] = feat
    for i in range(1: regr_x.shape[0]):
        imput_im[0] = normalize(im2feat(stimuli_train[i]))
        feat = np.squeeze(func(imput_im))
        regr_x = np.empty([1750, feat.shape[0]])
        regr_x[i] = feat

    regr = linear_model.LinearRegression()

    # X needs n samples and n features
    regr.fit(regr_x, response_train)

    # plt.subplot(2, 1, 1)
    # plt.imshow(stim_train[0], cmap='gray')
    # plt.title('Image')
    # plt.subplot(2, 1, 2)
    # plt.imshow(put_data_in_grid(feat[0]), cmap='gray', interpolation = 'nearest')
    # plt.title('Features')
    # plt.show()


if __name__ == '__main__':
    demo_brainreader()
