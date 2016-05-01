import numpy as np
import theano
from scipy.io import loadmat, savemat
import numpy as np
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd


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
