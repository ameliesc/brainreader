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

def demo_brainreader(sample_size = 1750, layer_name = 'pool1', regression = 'online'):
    ### get data ###
    
    stimuli_train, response_train = get_data(response=1)
    print response_train.shape
    input_im = np.empty([1, 3, 224, 224])
    input_im[0] = im2feat(stimuli_train[0])

    ### Convolution ###
    
    net = get_vgg_net(up_to_layer = layer_name)
    func = net.get_named_layer_activations.compile()
    named_features = func(input_im)
    feat = np.squeeze(named_features[layer_name + '_layer'])
    
    ### Regression ###
    print 'Begin Convolution training set...'
    regr_x = np.empty((sample_size, feat.shape[0] *feat.shape[1] * feat.shape[2])) # (1750, n_maps, size_y, size_x)
    regr_x[0] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))
    for i in range(1, sample_size):
        input_im[0] =im2feat(stimuli_train[i])
        named_features = func(input_im)
        feat = np.squeeze(named_features[layer_name + '_layer'])
        #pickle.dump(feat,open("featuremaps.p", 'w'))
        regr_x[i] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))
    print 'Finished.' 

    n_samples = sample_size
    n_features = regr_x.shape[1]
    y_train = response_train[0:sample_size,:]
    x_train = regr_x[:sample_size,:]
    
    stimuli_test, response_test = get_data(response = 1, data = 'test')

    print 'Begin Convolution test set...'
    regr_x_test = np.empty((stimuli_test.shape[0], feat.shape[0] *feat.shape[1] * feat.shape[2])) # (1750, n_maps, size_y, size_x)
    regr_x_test[0] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))
    
    for i in range(1, stimuli_test.shape[0]):
        input_im[0] =im2feat(stimuli_test[i])
        named_features = func(input_im)
        feat = np.squeeze(named_features[layer_name + '_layer'])
        #pickle.dump(feat,open("featuremaps.p", 'w'))
        regr_x_test[i] = np.reshape(feat,(feat.shape[0] *feat.shape[1] * feat.shape[2]))
    print 'Finished.'
    
    y_test = response_test
    x_test = regr_x_test
    a = 2.5e-4
    clf = KernelRidge(alpha = a)

    if regression == 'ridge':

    # ## Ridge Regression ##
        voxel_predictions = OrderedDict()
        voxel_coef = OrderedDict()
        voxel_model = OrderedDict()
        for i in range(0,1):
            trained = clf.fit(x_train, y_train[:,i])
            print clf.score(x_test, y_test[:,i])
            voxel_model[i] = clf
            voxel_predictions[i] = clf.predict(x_train)
            voxel_coef[i] = clf.dual_coef_
        return voxel_coef

    ## online regression ###
    if regression == 'online':
        
        predictor = LinearRegressor(x_train.shape[1], response_train.shape[1])
        f_train = predictor.train.compile()
        f_predict = predictor.predict.compile()
        training_data = regr_x
        test_data = regr_x_test
        training_target = y_train
        test_target =  y_test
        score_report_period = 100
        n_epochs = 2
        n_training_samples = x_train.shape[1]
        for i in xrange(n_training_samples*n_epochs+1):
            if i % score_report_period == 0:
                out = f_predict(test_data)
                test_cost = ((test_target-out)**2).sum(axis=1).mean(axis=0)
                print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
            f_train(training_data[[i % n_training_samples]], training_target[[i % n_training_samples]])

    
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
