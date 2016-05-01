import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from onlinergression_adams import LinearRegressor

def online_ridge():

    #['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1','conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                 #  'conv5_2',  'conv5_3',   'conv5_4'
    layer_names = ['fc6', 'fc7','fc8']

    regr_coef  = OrderedDict()
    regr_cost = OrderedDict()
    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
        print "Done."

        y_train = get_data(response=1, roi = 3)
        x_train = feature_map_train
        y_test = get_data(response=1, data='test', roi = 3 )  # n_samples x n_targets
        x_test = feature_map_test  # n_samples x n_feature
        batch_size = 10
        n_in = x_train.shape[1]
        n_out = batch_size
        n_training_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]
        score_report_period = 400
        n_epochs = 100
        lmbda = 0.001
        cost_voxel = np.zeros_like(y_test)
        weights_voxel = np.zeros((x_train.shape[1],y_train.shape[1]))
        j = 0

        
        # Train on one sample at a time and periodically report score.
  
        w_old = None
        j = 0
        while j < y_train.shape[1]:
            print "training batch "
            predictor = LinearRegressor(n_in, n_out, lmbda = lmbda)
            f_train = predictor.train.compile()
            f_predict = predictor.predict.compile()
            f_cost = predictor.voxel_cost.compile()
            test_cost_old = 0
            
            for i in xrange(0, n_training_samples*n_epochs+1):
                
                out = f_predict(x_test)
                test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0)
                    
                if np.isnan(test_cost) or np.isinf(test_cost):
                    print "Cost nan or inf resetting parameters."
                    alpha = predictor.get_params()
                    alphe = 0.5 * alpha
                    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, alpha = alpha)
                    f_train = predictor.train.compile()
                    f_predict = predictor.predict.compile()
                    f_cost = predictor.voxel_cost.compile()
                    print "new step size: %s" % (alpha)
                    continue

                if i % score_report_period == 0:
                    print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)

                f_train(x_train[[i % n_training_samples]], y_train[i % n_training_samples, j: j+batch_size])
                test_cost_old = test_cost
                
            cost_batch = f_cost(x_test, y_test[:,j:j+batch_size])
            cost_voxel[:,j:j+batch_size] =cost_batch
            w = predictor.coef_()
            w = w.get_value()
            weights_voxel[:,j:j+batch_size] = w
            j = j + batch_size
            
        regr_cost[name] = cost_voxel
        regr_coef[name] = weights_voxel
            
    dd.io.save("regression_coefficients_roi3_fc.h5", regr_coef)
    dd.io.save("regression_cost_roi3_fc.h5", regr_cost)
            #print f_cost(x_test,y_test) 
            #return f_cost(x_test, y_test) # return values for testing cost function in commandline
                #print "Cost train: %d" % (f_cost(x_train, y_train[:,i:i+10]))
                #print "Cost test: %d" % (f_cost(x_test,y_test[:,i:i+10]))
