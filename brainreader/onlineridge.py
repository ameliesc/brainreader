import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressiontheano import LinearRegressor

def online_ridge():

    #['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1','conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                 #  'conv5_2',  'conv5_3',  'conv5_4',
    layer_names = [  'fc6',  'fc7',
                   'fc8']

    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
        print "Done."

        y_train = get_data(response=1)
        x_train = feature_map_train
        y_test = get_data(response=1, data='test')  # n_samples x n_targets
        x_test = feature_map_test  # n_samples x n_feature

        n_in = x_train.shape[1]
        n_out = y_train.shape[1]
        n_training_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]
        i = 0
        score_report_period = 200
        n_epochs = 1
#0 2.7580e-25, 2.7198e-13, 6.5068e-08, 7.2153e+18,6.8359e+22,[   1.5407e+47, 4.5480e+51, 9.8656e+53, 7.9408e+58]
        lmbda_list = [0.001]
        for l in lmbda_list:

            predictor = LinearRegressor(n_in, n_out, lmbda = l, eta = 0.00000000000000000000000000000000000000000000000001)
            f_train = predictor.train.compile()
            f_predict = predictor.predict.compile()
            f_cost = predictor.cost.compile()
            # Train on one sample at a time and periodically report score.
            for i in xrange(n_training_samples*n_epochs+1):#
                #if i % score_report_period == 0:
                out = f_predict(x_test)
                w = predictor.coef_()
                w = w.get_value()
                test_cost = ((y_test - out)**2).sum(axis = 1).mean(axis=0) 
                print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
                if np.isnan(test_cost):
                    return w
                f_train(x_train[[i % n_training_samples]], y_train[[i % n_training_samples]])
            return w
            #print f_cost(x_test,y_test) 
            #return f_cost(x_test, y_test) # return values for testing cost function in commandline
                #print "Cost train: %d" % (f_cost(x_train, y_train[:,i:i+10]))
                #print "Cost test: %d" % (f_cost(x_test,y_test[:,i:i+10]))
