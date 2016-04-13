from brainreader.pretrained_networks import get_vgg_net
from brainreader.unwrap_deconvnet import get_deconv
from brainreader.makedeconvnet import load_conv_and_deconv
from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
from brainreader.art_gallery import get_image
import numpy as np
from matplotlib import pyplot as plt
import theano
from collections import OrderedDict

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
    deconv = load_conv_and_deconv()
    raw_content_image = get_image('starry_night', size=(224, 224))  # (im_size_y, im_size_x, n_colours)

    input_im = im2feat(raw_content_image)
    net = get_vgg_net(up_to_layer='pool1')

    func = net.get_named_layer_activations.compile()

    named_features = func(input_im)
    switch_dict = OrderedDict()
    for name in named_features:
        
        if 'switch' in name:
            switch_dict[name] = named_features[name]
    net = get_deconv(switch_dict, network_params=deconv, from_layer= 'pool1')

    func = net.compile()

    image_reconstruct = func(named_features['pool1_layer'])

    return image_reconstruct
