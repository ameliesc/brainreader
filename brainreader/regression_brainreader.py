import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from pretrained_networks import get_vgg_net
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
import h5py
from scipy.misc import imresize
from data_preprocessing import get_data
from general.ezprofile import EZProfiler

__author__ = 'amelie'


def im2feat(im):
    im = imresize(im, (224, 224))  #update to vggnet requirements ;incorporate normalize 
    im = np.transpose(np.dstack((im, im, im)))
    return im
# prepcossing of image :
# 1 smooth
# remove the mean


def feat2im(feat): return feat.dimshuffle(0, 2, 3, 1)[0, :, :, :]\
    if isinstance(feat, Variable) else np.rollaxis(feat, 0, 2)[0, :, :, :]


def normalize(arr): return (arr - np.mean(arr)) / np.std(arr)


def setup_net():
    net = get_vgg_net(up_to_layer='pool2')
    # See function get_vgg_net for the layers that you can
    func = net.compile()
    return func


def demo_brainreader():
    stimuli_train, response_train = get_data(response=1)
    with EZProfiler(profiler_name = 'init-time'):
        func = setup_net()
        imput_im = np.empty([1, 3, 224, 224])
        imput_im[0] = normalize(im2feat(stimuli_train[0]))
        feat = np.squeeze(func(imput_im))
        regr_x = np.empty((1750, feat.shape[0] *feat.shape[1] * feat.shape[2]) # (1750, n_maps, size_y, size_x)
        regr_x[0] = imresize(feat
    for i in range(1, regr_x.shape[0]):
        with EZProfiler(profiler_name = 'lap-time'):
            imput_im[0] = normalize(im2feat(stimuli_train[i]))
            feat = np.squeeze(func(imput_im))
            pickle.dump(feat,open("featuremaps.p", 'w'))
            #regr_x = np.empty([1750, feat.shape[0]])
            regr_x[i] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))

    regr = linear_model.LinearRegression()

    # X needs n samples and n features
    regr_model = regr.fit(regr_x, response_train)
    pickle.dump(regr_model, open("regressionvalues.p", "w"))

    # plt.subplot(2, 1, 1)
    # plt.imshow(stim_train[0], cmap='gray')
    # plt.title('Image')
    # plt.subplot(2, 1, 2)
    # plt.imshow(put_data_in_grid(feat[0]), cmap='gray', interpolation = 'nearest')
    # plt.title('Features')
    # plt.show()


if __name__ == '__main__':
    demo_brainreader()
