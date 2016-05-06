import matplotlib
matplotlib.use('Agg')
from pretrained_networks import get_vgg_net
from unwrap_deconvnet import get_deconv
from makedeconvnet import load_conv_and_deconv
from plotting.data_conversion import put_data_in_grid
from theano.gof.graph import Variable
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
import deepdish as dd
from matplotlib.backends.backend_pdf import PdfPages
from analyse import filtervoxels

def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR image that's ready to feed into VGGNet

    """
    centered_bgr_im = im[:, :, ::-1] - np.array([103.939, 116.779, 123.68])
    feature_map_im = np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    return feature_map_im.astype(theano.config.floatX)

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

def conv_and_deconv(layername, n, voxel_index):

    #raw_content_image = get_image('trump', size=(224, 224))  # (im_size_y, im_size_x, n_colours)
    #input_im = im2feat(raw_content_image)
   
    net = get_vgg_net(up_to_layer = layername)
    func = net.get_named_layer_activations.compile()
    stimuli_test = get_data(data='test')
    input_im = np.empty([1, 3, 224, 224])
    input_im = im2feat(stimuli_test[0])
    print input_im.shape
    named_features = func(input_im)

    switch_dict = OrderedDict()
    for name in named_features:
        
        if 'switch' in name:
            switch_dict[name] = named_features[name]

    features =  named_features[layername+'_layer']

    weights = dd.io.load('/data/regression_coefficients_roi%s_%s.h5' % (n, layername))
    w_times_feat = features * np.reshape(weights[:, voxel_index],features.shape)
    features = w_times_feat
    deconv = load_conv_and_deconv()
    net = get_deconv(switch_dict, network_params=deconv, from_layer= layername)
    func = net.compile()
    image_reconstruct = func(features)
    raw_content_image = feat2im(im2feat(stimuli_test[0]))
    return image_reconstruct, raw_content_image
    #maxval = np.amax(image_reconstruct, axis = 1)
    #zeroed = np.asarray(image_reconstruct)
    #indices = zeroed < maxval
    #zeroed[indices] = 0
    #zeroed
     # Plot

   


    
def layer_images():
    for layername in ['fc6', 'fc7', 'fc8']:
         for  i in [7,6,2,1]:
            print "reconstructing image for layer %s and region %s" % (layername, i)
            pp = PdfPages('%s_%s.pdf' % (layername, i))
            dic  = filtervoxels(layername,n = i)
            cost = dic[1][0]
            index = dic[1][1][0]
            index_1 = np.where(cost < 10)
            index = index[index_1]
            net = get_vgg_net(up_to_layer = layername)
            conv = net.get_named_layer_activations.compile()
            for j in range(0,index.shape[0]):
                for k in range(0,120): 
                    stimuli_test = get_data(data='test')
                    input_im = np.empty([1, 3, 224, 224])
                    input_im = im2feat(stimuli_test[k])
                    print input_im.shape
                    named_features = conv(input_im)

                    switch_dict = OrderedDict()
                    for name in named_features:

                        if 'switch' in name:
                            switch_dict[name] = named_features[name]

                    features =  named_features[layername+'_layer']

                    weights = dd.io.load('/data/regression_coefficients_roi%s_%s.h5' % (i, layername))
                    w_times_feat = features * np.reshape(weights[:, j],features.shape)
                    features = w_times_feat
                    deconv = load_conv_and_deconv()
                    net = get_deconv(switch_dict, network_params=deconv, from_layer= layername)
                    deconv = net.compile()
                    image_reconstructed = deconv(features)
                    raw_content_image = feat2im(im2feat(stimuli_test[k]))
                    plt.figure(figsize = (8,3))
                    plt.subplot(2, 1, 1)
                    plt.imshow(raw_content_image, cmap='Greys_r')
                    plt.title('Original Image')
                    plt.subplot(2, 1, 2)
                    plt.imshow(feat2im(image_reconstructed), cmap='Greys_r')
                    plt.title('Reconstuction of voxel %s' % (j))
                    pp.savefig()
            pp.close()
