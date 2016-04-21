import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
from sklearn.kernel_ridge import KernelRidge
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
    for name in layer_names:
        feature_map_train = dd.io.load("featurmap_train_%s" % (name))
        feature_map_test = dd.io.load("featuremap_%s" % (name))
        n_samples = sample_size
        n_features = regr_x.shape[1]
        y_train = get_data(response = 1)
        x_train = feature_map_train
        y_test = get_data(response = 1, data = 'test')
        x_test = feature_map_test
        a = 2.5e-4
        clf =  GridSearchCV(KernelRidge(alpha = a),cv=5,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3, 1e-4, 2.5e-4],
                               "gamma": np.logspace(-2, 2, 0.8, 5, -5)})
 
        print "Training using feature map: %s" % (name)
        trained = clf.fit(x_train, y_train)
        print clf.score(x_test, y_test)
        predict = clf.predict(x_test)
        voxel_predictions[name] = predict
        best_params[name] = clf.best_params_
        voxel_coef[name] = clf.dual_coef_  

    print "Saving coefficients..."
    dd.io.save("voxel_coef.h5", voxel_coef)
    print "Saving predictions..."
    dd.io.save("voxel_predictions.h5", voxel_predictions)
    print "Done."
    dd.io.save("best_params.h5", best_params)
    return voxel_coef, voxel_predictions, best_params
