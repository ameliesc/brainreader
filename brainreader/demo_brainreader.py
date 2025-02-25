from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
from brainreader.art_gallery import get_image
from brainreader.pretrained_networks import get_vgg_net
import numpy as np
from matplotlib import pyplot as plt
import theano

__author__ = 'peter'


def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet

    """
    centered_bgr_im = im[:, :, ::-1] - np.array([103.939, 116.779, 123.68])
    feature_map_im = np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    return feature_map_im.astype(theano.config.floatX)


def feat2im(feat):
    """
    :param feat: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet
    :returns: A (size_y, size_x, 3) array representing a RGB image.
    """
    bgr_im = np.rollaxis(feat, 0, 2)[0, :, :, :]
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im


def demo_brainreader():

    # Need to get dataset.  See from scipy.io import loadmat.  Example in pretrained_networks.py may be useful.

    raw_content_image = get_image('starry_night', size=(224, 224))  # (im_size_y, im_size_x, n_colours)

    input_im = im2feat(raw_content_image)  # (n_samples, n_colours, im_size_y, im_size_x)  -- where n_samples=1 and n_colours=3
    net = get_vgg_net(up_to_layer='pool1')  # See function get_vgg_net for the layers that you can go up to.
    func = net.get_named_layer_activations.compile()# Compile the network into a function that takes input_im and returns features.  #TODO: add functionality for outputting multiple feature layers
    
    named_features = func(input_im)  # dict<layer_name: (n_samples, n_feat_maps, feat_size_y, feat_size_x) array> -- where n_samples=1, layer_name is a string
    return named_features

    # # Plot
    # while True:
    #     layer = raw_input('Layers: %s\nType Layer to view >> ' % (named_features.keys()))
    #     if layer not in named_features:
    #         print 'Error: No layer named %s' % (layer, )
    #     else:
    #         plt.subplot(2, 1, 1)
    #         plt.imshow(raw_content_image)
    #         plt.title('Image')
    #         plt.subplot(2, 1, 2)
    #         plt.imshow(put_data_in_grid(named_features[layer][0]), cmap='gray', interpolation = 'nearest')
    #         plt.title('Features')
    #         plt.show()

    # You can use sklearn for Linear Regression.

    # For the deconv stuff, I have a similar project that does gradient descent on the image (it's not exactly deconvolution
    # but similar) example code in:
    # https://github.com/petered/liquid-style/blob/move-on-over/liquid_style/demo_liquid_style.py


