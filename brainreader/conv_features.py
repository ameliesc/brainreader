from pretrained_networks import get_vgg_net
from theano.gof.graph import Variable
import numpy as np
import theano
from scipy.misc import imresize
from data_preprocessing import get_data
from collections import OrderedDict
from scipy.io import loadmat, savemat
import cPickle as pickle
import deepdish as dd


def im2feat(im):
    """
    :param im: A (size_y, size_x, 3) array representing a RGB image on a [0, 255] scale
    :returns: A (1, 3, size_y, size_x) array representing the BGR
    image that's ready to feed into VGGNet
cat
    """
    im = imresize(
        im, (224, 224))  # update to vggnet requirements ;incorporate normalize
    im = (np.dstack((im, im, im)))
    centered_bgr_im = im[:, :, ::-1] - np.array([103.939, 116.779, 123.68])
    feature_map_im = np.rollaxis(centered_bgr_im, 2, 0)[None, :, :, :]
    return feature_map_im.astype(theano.config.floatX)


def get_featuremaps(sample_size=120, layer_name=None, data_set='test'):

    ### get data ###
    if data_set == 'train':
        stimuli_train = get_data()
    else:
        stimuli_train = get_data(data='test')
    input_im = np.empty([1, 3, 224, 224])
    input_im[0] = im2feat(stimuli_train[0])


    ####  Case get all layers 
    if layer_name == None:

        layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1',
                       'conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                       'conv5_2',  'conv5_3',  'conv5_4',  'fc6',  'fc7',
                       'fc8']

        stim = im2feat(stimuli_train[0])
        input_im = stim

        feature_maps = OrderedDict()
        for l_name in layer_names:
            
            
            net = get_vgg_net(up_to_layer = l_name)
            func = net.get_named_layer_activations.compile()
            input_im =  im2feat(stimuli_train[0])
            named_features = func(input_im)
            feat = named_features[l_name + '_layer']
            regr_x = np.empty((sample_size, feat.shape[1] * feat.shape[2] * feat.shape[3]))
            regr_x[0] = np.reshape(feat, (feat.shape[1] * feat.shape[2] * feat.shape[3]))

            print 'Convolving images up to layer %s ...' % (l_name)
            for i in range(1, sample_size):

                input_im =  im2feat(stimuli_train[i])
                named_features = func(input_im)
                feat = named_features[l_name + '_layer']
                regr_x[0] = np.reshape(feat, (feat.shape[1] * feat.shape[2] * feat.shape[3]))

            feature_maps[l_name] = regr_x
            print "Done."

            print "Saving feature_maps..."    
            dd.io.save("featuremap_%s" % l_name,regr_x)
        print "Done."
        return feature_maps



if __name__ == '__main__':
   get_featuremaps()
