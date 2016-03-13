from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
from brainreader.art_gallery import get_image
from brainreader.pretrained_networks import get_vgg_net
import numpy as np
from matplotlib import pyplot as plt

__author__ = 'peter'


def im2feat(im): return im.dimshuffle('x', 2, 0, 1) if isinstance(im, Variable) else np.rollaxis(im, 2, 0)[None, :, :, :]

def feat2im(feat): return feat.dimshuffle(0, 2, 3, 1)[0, :, :, :] if isinstance(feat, Variable) else np.rollaxis(feat, 0,2)[0, :, :, :]

def normalize(arr): return (arr - np.mean(arr)) / np.std(arr)


def demo_brainreader():

    # Need to get dataset.  See from scipy.io import loadmat.  Example in pretrained_networks.py may be useful.

    raw_content_image = get_image('starry_night', size=(128, None))  # (im_size_y, im_size_x, n_colours)

    input_im = normalize(im2feat(raw_content_image))  # (n_samples, n_colours, im_size_y, im_size_x)  -- where n_samples=1 and n_colours=3
    print input_im.shape
    net = get_vgg_net(up_to_layer='pool4')  # See function get_vgg_net for the layers that you can go up to.
    func = net.compile()  # Compile the network into a function that takes input_im and returns features.  #TODO: add functionality for outputting multiple feature layers
    feat = func(input_im)  # shape (n_samples, n_feat_maps, feat_size_y, feat_size_x) -- where n_samples=1

    # Plot
    plt.subplot(2, 1, 1)
    plt.imshow(raw_content_image)
    plt.title('Image')
    plt.subplot(2, 1, 2)
    plt.imshow(put_data_in_grid(feat[0]), cmap='gray', interpolation = 'nearest')
    plt.title('Features')
    plt.show()

    # You can use sklearn for Linear Regression.

    # For the deconv stuff, I have a similar project that does gradient descent on the image (it's not exactly deconvolution
    # but similar) example code in:
    # https://github.com/petered/liquid-style/blob/move-on-over/liquid_style/demo_liquid_style.py


if __name__ == '__main__':
    demo_brainreader()
