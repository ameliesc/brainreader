from brainreader.pretrained_networks import get_vgg_net
from brainreader.unwrap_deconvnet import get_deconv
from brainreader.makedeconvnet import load_conv_and_deconv
from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
from brainreader.art_gallery import get_image
import numpy as np
from matplotlib import pyplot as plt
import theano
import h5py
from scipy.misc import imresize
from data_preprocessing import get_data
from collections import OrderedDict
from plotting.data_conversion import put_data_in_grid
from sklearn import linear_model
from scipy.io import loadmat
from matplotlib import pyplot as plt

def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet

    """
    im = imresize(im, (224, 224))  #update to vggnet requirements ;incorporate normalize 
    im = (np.dstack((im, im, im)))
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

def demo_brainreader(sample_size = 1750, layer_name = 'pool1'):
    ### get data ###
    
    stimuli_train, response_train = get_data(response=1)
    input_im = np.empty([1, 3, 224, 224])
    input_im[0] = im2feat(stimuli_train[0])

    ### Convolution ###
    
    net = get_vgg_net(up_to_layer = layer_name)
    func = net.get_named_layer_activations.compile()
    named_features = func(input_im)
    feat = np.squeeze(named_features[layer_name + '_layer'])
    
    ### Regression ###
    
    regr_x = np.empty((sample_size, feat.shape[0] *feat.shape[1] * feat.shape[2])) # (1750, n_maps, size_y, size_x)
    regr_x[0] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))
    for i in range(1, sample_size):
        input_im[0] =im2feat(stimuli_train[i])
        named_features = func(input_im)
        feat = np.squeeze(named_features[layer_name + '_layer'])
        print feat.shape
        print (sample_size, feat.shape[0] *feat.shape[1] * feat.shape[2])
        #pickle.dump(feat,open("featuremaps.p", 'w'))
        regr_x[i] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))

    regr = linear_model.LinearRegression()
    # X needs n samples and n features
    regr_model = regr.fit(regr_x, response_train[:sample_size])

    ### Deconvolution 
    switch_dict = OrderedDict()
    for name in named_features:
        
        if 'switch' in name:
            switch_dict[name] = named_features[name]

    deconv = load_conv_and_deconv()
    net = unwrap_deconvnet(switch_dict, network_params=deconv, from_layer= layer_name)
    func = net.compile()
    image_reconstruct = func( np.reshape(regr_model.coef_,feat) * feat)

    ### Plotting ###
    plt.subplot(2, 1, 1)
    plt.imshow(raw_content_image)
    plt.title('Image')
    plt.subplot(2, 1, 2)
     # plt.imshow(put_data_in_grid(named_features[layer][0]),
     #cmap='gray', interpolation = 'nearest')
    plt.imshow(feat2im(image_reconstruct, cmap = 'gray'))
    plt.title('Features')
    plt.show()

    return image_reconstruct
