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
from sklearn.kernel_ridge import KernelRidge
from regressiontheano import LinearRegressor
from regression_features import get_featuremaps
import cPickle as pickle


def demo_brainreader(sample_size = 1750):
    ### get data ###

    layer_names = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
                       'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1',
                       'conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                       'conv5_2',  'conv5_3',  'conv5_4',  'fc6',  'fc7',
                       'fc8']

    voxel_predictions = OrderedDict()
    voxel_predictions[0] = 0
    voxel_coef = OrderedDict()
    voxel_coef[0] = 0
    voxel_model = OrderedDict()
    voxel_model[0] = 0:
    

    for name in layer_names:

        with open(r"train_%s.pickle" % (name), "rb") as input_file:
            feature_map_train = cPickle.load(input_file)
        with open(r"test_%s.pickle" % (name), "rb") as input_file:
            feature_map_test = cPickle.load(input_file)

        regr_x = feature_map_train
        regr_x_test = feature_map_test
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
        voxel_model[i] = clf
        predict = clf.predict(x_test)
        voxel_predictions[name] = predict
        voxel_coef[name] = clf.dual_coef_  #n_target, #n_features
        return voxel_coef, voxel_predictions

   

    
    # ### Deconvolution 
    # switch_dict = OrderedDict()
    # for name in named_features:
        
    #     if 'switch' in name:
    #         switch_dict[name] = named_features[name]

    # deconv = load_conv_and_deconv()
    # net = unwrap_deconvnet(switch_dict, network_params=deconv, from_layer= layer_name)
    # func = net.compile()
    # image_reconstruct = func( np.reshape(predictor.coef_,feat) * feat)

    # ### Plotting ###
    # plt.subplot(2, 1, 1)
    # plt.imshow(raw_content_image)
    # plt.title('Image')
    # plt.subplot(2, 1, 2)
    #  # plt.imshow(put_data_in_grid(named_features[layer][0]),
    #  #cmap='gray', interpolation = 'nearest')
    # plt.imshow(feat2im(image_reconstruct, cmap = 'gray'))
    # plt.title('Features')
    # plt.show()

    # return image_reconstruct
