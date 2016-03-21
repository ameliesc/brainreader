import numpy as np
from scipy.io import loadmat
from matplotlib import pyplot as plt
from pretrained_networks import get_vgg_net
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
import h5py
from scipy.misc import imresize

__author__ = 'amelie'


def im2feat(im):
    im = imresize(im, (256, 256))
    np.transpose (np.dstack((im, im, im)))
    return im
#prepcossing of image :
## 1 smooth
## remove the mean 
def feat2im(feat): return feat.dimshuffle(0, 2, 3, 1)[0, :, :, :] if isinstance(feat, Variable) else np.rollaxis(feat, 0,2)[0, :, :, :]

def normalize(arr): return (arr - np.mean(arr)) / np.std(arr)


def demo_brainreader():
    

    stimuli = loadmat('Stimuli.mat')
    stim_train = stimuli['stimTrn'] #size 1750 x 128 x 128
    stim_valid = stimuli['stimVal']

    # Reshape stimuli greyscale into rgb
    stimuli = np.empty([1750,3,256,256])
    for i in range (0,stim_train.shape[0]):

        stimuli[i] = normalize(im2feat(stim_train[i]))

    net = get_vgg_net(up_to_layer='fc8')  # See function get_vgg_net for the layers that you can go up to.
    func = net.compile()  # Compile the network into a function that takes input_im and returns features.  #TODO: add functionality for outputting multiple feature layers
    imput_im = np.empty([1,3,244,244])
    imput_im[0] = stimuli[0]
    feat = func(imput_im)  # shape (n_samples, n_feat_maps, feat_size_y, feat_size_x) -- where n_samples=1
    print feat.shape
    response = h5py.File('EstimatedResponses.mat')
    response_train = response['dataTrnS1']
    regr = linear_model.LinearRegression()
    target_voxel_i = response_train[0:1740][0]
    regr.fit(feat, target_voxel_i)

    # plt.subplot(2, 1, 1)
    # plt.imshow(stim_train[0], cmap='gray')
    # plt.title('Image')
    # plt.subplot(2, 1, 2)
    # plt.imshow(put_data_in_grid(feat[0]), cmap='gray', interpolation = 'nearest')
    # plt.title('Features')
    # plt.show()


if __name__ == '__main__':
    demo_brainreader()
