import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressiontheano import LinearRegressor

def online_ridge():

    #['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1','conv4_2', 'conv4_3',  'conv4_4', 'conv5_1',
                 #  'conv5_2',  'conv5_3',   'conv5_4', 'fc6',
    layer_names = [ 'fc7','fc8']

    regr_coef  = OrderedDict()
    regr_cost = OrderedDict()
    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
        print "Done."

        roi = 4
        y_train = get_data(response=1, roi = roi)
        x_train = feature_map_train       
        x_test = feature_map_test  # n_samples x n_feature
        y_test = get_data(response=1, data='test', roi = roi)  # n_samples x n_targets
        y_test = y_test[:,0:y_train.shape[1]] #test dimensions not always the same, due to discarded nan voxels
        batch_size = 100
        n_in = x_train.shape[1]
        n_out = y_train.shape[1]

        n_training_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]
        i = 0
        score_report_period = 400
        n_epochs = 2
        lmbda = 0.01

        predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = 0.01 )
        f_train = predictor.train.compile()
        f_predict = predictor.predict.compile()
        f_cost = predictor.voxel_cost.compile()
        f_cost(x_test,y_test)
        # Train on one sample at a time and periodically report score.
        epoch = 0
        test_cost_old = 0
        w_old = None
        
        while i < n_training_samples*n_epochs+1:
            out = f_predict(x_test)
            test_cost = ((y_test - out)**2).sum(axis = 1).mean(axis=0) 
            #skip NaN reslts
            if i % score_report_period == 0:
                print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
            if np.isnan(test_cost) or np.isinf(test_cost):
                print "Cost nan or inf (%s) resetting parameters." % (test_cost)
                i = 0
                eta = predictor.get_params()
                eta = eta * 0.5
                predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old )
                f_train = predictor.train.compile()
                f_predict = predictor.predict.compile()
                f_cost = predictor.voxel_cost.compile()
                print "new step size: %s" % (eta)
                continue
            

            if i == epoch: # Adaptive Stepsize
 
                if test_cost_old < test_cost:
                    print "Cost too high (%s))" % (test_cost)
                    eta = predictor.get_params()
                    eta = 0.5 * eta
                    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old )
                    f_train = predictor.train.compile()
                    f_predict = predictor.predict.compile()
                    f_cost = predictor.voxel_cost.compile()
                    print "Decreasing  learning rate: %s" % (eta)
                    #set_params(alpha = alpha)

                elif np.isnan(test_cost) or np.isinf(test_cost):
                    print "Cost nan or inf (%s) resetting parameters." % (test_cost)
                    eta = predictor.get_params()
                    eta = 0.5*eta
                    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old )
                    f_train = predictor.train.compile()
                    f_predict = predictor.predict.compile()
                    f_cost = predictor.voxel_cost.compile()
                    print "Decreasing learning rate: %s" % (eta)
                    i = 0

                elif test_cost_old > test_cost or test_cost_old == test_cost:
                    eta = predictor.get_params()
                    eta_old = eta
                    eta = 1.1 * eta
                    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old )
                    f_train = predictor.train.compile()
                    f_predict = predictor.predict.compile()
                    f_cost = predictor.voxel_cost.compile()
                    
                    print "Increasing learning rate from %s to  %s" % (eta_old, eta)
                        
                
                    # predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = alpha*eta, w=  w_old )# decrease learning rate by alpha
                    # f_train = predictor.train.compile()
                    # f_predict = predictor.predict.compile()
                epoch += n_training_samples

                test_cost_old = test_cost
                w = predictor.coef_()
                w_old = w.get_value()
            f_train(x_train[[i % n_training_samples]], y_train[[i % n_training_samples]])
            i += 1

                
        w = predictor.coef_()
        w_old = w.get_value()
        cost_voxel = f_cost(x_test,y_test)
        regr_cost["name"] = cost_voxel
        regr_coef["name"] =  w_old
            
    dd.io.save("regression_coefficients_roi%s_%s.h5" % (roi,name), regr_coef)
    dd.io.save("regression_cost_roi%s_%s.h5" % (roi,name), regr_cost)
            #print f_cost(x_test,y_test) 
            #return f_cost(x_test, y_test) # return values for testing cost function in commandline
                #print "Cost train: %d" % (f_cost(x_train, y_train[:,i:i+10]))
                #print "Cost test: %d" % (f_cost(x_test,y_test[:,i:i+10]))
