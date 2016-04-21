from pretrained_networks import get_vgg_net
from theano.gof.graph import Variable
import numpy as np
import theano
from scipy.misc import imresize
from data_preprocessing import get_data
from collections import OrderedDict
from scipy.io import loadmat, savemat
import cPickle as pickle

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
            print feat.shape
            regr_x = np.empty((sample_size, feat.shape[1] * feat.shape[2] * feat.shape[3]))
            regr_x[0] = np.reshape(feat, (feat.shape[1] * feat.shape[2] * feat.shape[3]))

            print 'Convolving images up to layer %s' % (l_name)
            for i in range(1, sample_size):

                input_im =  im2feat(stimuli_train[i])
                named_features = func(input_im)
                feat = named_features[l_name + '_layer']
                regr_x[0] = np.reshape(feat, (feat.shape[1] * feat.shape[2] * feat.shape[3]))
           # print "Saving Feature maps to matlab file and pickle..."

            #with open("train_%s.pickle" % (l_name), "wb") as output_file:
               # pickle.dump(regr_x, output_file)
            feature_maps[l_name] = regr_x
            print "Done"
            return feature_maps

def kernel_ridge():
    print "Getting feature maps for training..."
    feature_map_train = get_featuremaps(sample_size = 1750, data_set = 'train')
    print "Getting feature maps for test..."
    feature_map_test = get_featuremaps(sample_size = 120, data_set = 'test')
    print "Done."
    voxel_predictions = OrderedDict()
    voxel_predictions[0] = 0
    voxel_coef = OrderedDict()
    voxel_coef[0] = 0
    for name in layer_names:

        regr_x = feature_map_train[name]
        regr_x_test = feature_map_test[name]
        n_samples = sample_size
        n_features = regr_x.shape[1]
        response_train = get_data(response = 1)
        y_train = response_train[0:sample_size,:]
        x_train = regr_x[:sample_size,:]
    
        response_test = get_data(response = 1, data = 'test')
        y_test = response_test
        x_test = regr_x_test
        a = 2.5e-4
        clf =  GridSearchCV(KernelRidge(alpha = a),cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e-4, 2.5e-4],
                               "gamma": np.logspace(-2, 2, 5)})
 


        print "feature map: %s" % (name)
        trained = clf.fit(x_train, y_train)
        print clf.score(x_test, y_test)
        predict = clf.predict(x_test)
        voxel_predictions[name] = predict
        voxel_coef[name] = clf.dual_coef_  

    print "Saving coefficients"
    with open("voxel_coefficients.pickle", "wb") as output_file:
                pickle.dump(voxel_coef, output_file, protocol=pickle.HIGHEST_PROTOCOL )
    print "Saving predictions"
    with open("voxel_predictions.pickle", "wb") as output_file:
                pickle.dump(voxel_predictions, output_file, protocol=pickle.HIGHEST_PROTOCOL)
    return voxel_coef, voxel_predictions



    # else:
    #     feat = np.squeeze(named_features[layer_name + '_layer'])
    #     # (1750, n_maps, size_y, size_x)
    #     regr_x = np.empty(
    #         (sample_size, feat.shape[0] * feat.shape[1] * feat.shape[2]))
    #     regr_x[0] = np.reshape(
    #         feat, (feat.shape[0] * feat.shape[1] * feat.shape[2]))
    #     for i in range(1, sample_size):
    #         input_im[0] = im2feat(stimuli_train[i])
    #         named_features = func(input_im)
    #         feat = np.squeeze(named_features[layer_name + '_layer'])
    #         regr_x[i] = np.reshape(
    #             feat, (feat.shape[0] * feat.shape[1] * feat.shape[2]))
    #     feature_maps = regr_x
    # print 'Finished.'
    return feature_maps

if __name__ == '__main__':
   kernel_ridge()
