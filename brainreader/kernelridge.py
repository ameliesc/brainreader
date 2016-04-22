import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
from sklearn.kernel_ridge import KernelRidge
from sklearn.grid_search import GridSearchCV
import deepdish as dd


def kernel_ridge():

    # print "Getting feature maps for training..."
    # feature_map_train = get_featuremaps(sample_size = 1750, data_set = 'train')
    # print "Getting feature maps for test..."
    # feature_map_test = get_featuremaps(sample_size = 120, data_set = 'test')
    # print "Done."
    # voxel_prediction

    voxel_predictions = OrderedDict()
    voxel_predictions[0] = 0
    voxel_coef = OrderedDict()
    voxel_coef[0] = 0
    best_params = OrderedDict()
    best_params[0] = 0
    #['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1',
    layer_names = ['conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                   'conv5_2',  'conv5_3',  'conv5_4',  'fc6',  'fc7',
                   'fc8']

    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremap_%s" % (name))
        print "Done."

        y_train = get_data(response=1)
        x_train = feature_map_train
        print y_train.shape
        print x_train.shape
        y_test = get_data(response=1, data='test')  # n_samples x n_targets
        x_test = feature_map_test  # n_samples x n_feature
        clf = KernelRidge(alpha=2.5e-4, gamma=3)

        # glf = GridSearchCV(clf, cv=3, param_grid={"alpha": [1e0, 1e1, 1e2, 1e3, 1e-4, 2.5e-4],
        #                                          "gamma": np.logspace(-2, 2, 5)}, n_jobs=-1, pre_dispatch=8)
        predict = np.zeros(y_test.shape)
        coef = np.zeros((y_train.shape[1], x_train.shape[1],))
        print "Training using feature map: %s" % (name)
        for i in range(800, y_train.shape[1]):

            # X [n_samples, n_features], y [n_samples]
            clf.fit(x_train, y=y_train[:, i])
            print clf.dual_coef_.shape
            predict[:, i] = clf.predict(x_test)
            print "score train: %d" % (clf.score(x_train, y_train[:, i]))
            print "score test: %d" % (clf.score(x_test, y_test[:, i]))
            # coef[i,:] = clf.dual_coef_  #n_targets x n_features
        voxel_predictions[name] = predict
        #best_params[name] = clf.best_params_
        voxel_coef[name] = coef

    print "Saving coefficients..."
    dd.io.save("voxel_coef.h5", voxel_coef)
    print "Saving predictions..."
    dd.io.save("voxel_predictions.h5", voxel_predictions)
    return voxel_coef, voxel_predictions
