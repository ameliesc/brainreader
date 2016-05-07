import matplotlib
matplotlib.use('Agg')
from pretrained_networks import get_vgg_net
from unwrap_deconvnet import get_deconv
from makedeconvnet import load_conv_and_deconv
from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
from art_gallery import get_image
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
from matplotlib.backends.backend_pdf import PdfPages

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
    print feat.shape
    #bgr_im = np.rollaxis(feat, 0, 2)[0, :, :, :] ## doesnt work outputs (3,1,224,224)
    bgr_im = np.swapaxes(feat,0,2)
    bgr_im = np.swapaxes(bgr_im,1,3)
    bgr_im = np.swapaxes(bgr_im,2,3)
    bgr_im = bgr_im[:,:,:,0]
    print bgr_im.shape
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im

def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR
    image that's ready to feed into VGGNet
cat
    """
    im = imresize(im, (224, 224))  # update to vggnet requirements ;incorporate normalize
    im = (np.dstack((im, im, im)))
    centered_bgr_im = im[:, :, ::-1] - np.array([103.939, 116.779, 123.68])
    feature_map_im = np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    return feature_map_im.astype(theano.config.floatX)

def feat2im(feat):
    """
    :param feat: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet
    :returns: A (size_y, size_x, 3) array representing a RGB image.
    """
    print feat.shape
    #bgr_im = np.rollaxis(feat, 0, 2)[0, :, :, :] ## doesnt work outputs (3,1,224,224)
    bgr_im = np.swapaxes(feat,0,2)
    bgr_im = np.swapaxes(bgr_im,1,3)
    bgr_im = np.swapaxes(bgr_im,2,3)
    bgr_im = bgr_im[:,:,:,0]
    print bgr_im.shape
    decentered_rgb_im = (bgr_im + np.array([103.939, 116.779, 123.68]))[:, :, ::-1]
    return decentered_rgb_im



def demo_brainreader(layername):
    stimuli_test = get_data(data='test')
    for i in range(0,10):
        
        input_im = np.empty([1, 3, 224, 224])
        input_im = im2feat(stimuli_test[i])
        raw_content_image = feat2im(im2feat(stimuli_test[i]))

        net = get_vgg_net(up_to_layer = layername)
        conv = net.get_named_layer_activations.compile()
        named_features = conv(input_im)


        switch_dict = OrderedDict()
        for name in named_features:

            if 'switch' in name:
                switch_dict[name] = named_features[name]

        deconv_net = load_conv_and_deconv()
        net = get_deconv(switch_dict, network_params=deconv_net, from_layer= layername)
        deconv = net.compile()
        max_act = np.amax(named_features[layername+'_layer'], axis = 1)
        zeroed = np.asarray(named_features[layername+'_layer'])
        indices = zeroed < max_act
        zeroed[indices] = 0
        image_reconstruct = deconv(zeroed)
        maxval = np.amax(image_reconstruct, axis = 1)

         # Plot
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(raw_content_image)
        plt.title('Image')
        plt.subplot(2, 1, 2)
         # plt.imshow(put_data_in_grid(named_features[layer][0]),
         #cmap='gray', interpolation = 'nearest')
        plt.imshow(feat2im(image_reconstruct))
        plt.title('Features stronges activation')
        plt.show()
        plt.savefig('%s_wo_weights_image%s_max_act.png' % (layername,i))
