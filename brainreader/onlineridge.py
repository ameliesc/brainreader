import numpy as np
import h5py
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressiontheano import LinearRegressor

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

        y_train = get_data(response=1)
        x_train = feature_map_train
        y_test = get_data(response=1, data='test')  # n_samples x n_targets
        x_test = feature_map_test  # n_samples x n_feature
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
        set_params = predictor.set_params.compile()
        f_cost = predictor.voxel_cost.compile()
        f_cost(x_test,y_test)
        # Train on one sample at a time and periodically report score.
        epoch = 0
        test_cost_old = 0
        roh = 1.1
        alpha = 0.3
        w_old = None
        
        while i < n_training_samples*n_epochs+1:
            out = f_predict(x_test)
            test_cost = ((y_test - out)**2).sum(axis = 1).mean(axis=0) 
            #skip NaN reslts
            if test_cost_old < test_cost:
                test_cost = test_cost
            elif test_cost_old > test_cost or test_cost_old == test_cost:
                test_cost = test_cost
            else:
                print "Cost nan or inf resetting parameters."
                
                # predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = alpha*eta, w=  w_old )
                # f_train = predictor.train.compile()
                # f_predict = predictor.predict.compile()
                predictor.set_params(alpha=alpha)
                i = 0
                epoch = 0
                eta = predictor.get_params()
                eta = eta.get_value()
                print "new step size: %s" % (eta)
            if i % score_report_period == 0:
                print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
            eta = predictor.get_params()
            if i == epoch: # Adaptive Stepsize
 
                if test_cost_old < test_cost:
                   
                    #predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta.eval(), w=  w_old ) # decrease learning rate by alpha
                    #f_train = predictor.train.compile()
                    #f_predict = predictor.predict.compile()
                    #
                    print "Cost too high resetting parameters."
                    eta = predictor.get_params()
                    eta = eta.get_value()
                    predictor.set_params(alpha=alpha, eta = eta, w = w_old)
                    print "new step size: %s" % (eta)
                    #set_params(alpha = alpha)

                elif test_cost_old > test_cost or test_cost_old == test_cost:
                    eta = predictor.get_params()
                    eta = eta.get_value()
                    predictor.set_params(roh = roh, eta = eta)
                    print "Current learning rate: %s" % (eta)
                        
                else:
                    print "Cost nan or inf resetting parameters."
                    eta = predictor.get_params()
                    eta = eta.get_value()
                    predictor.set_params(alpha=alpha, eta = eta, w = w_old)
                    print "new step size: %s" % (eta)
                    # predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = alpha*eta, w=  w_old )# decrease learning rate by alpha
                    # f_train = predictor.train.compile()
                    # f_predict = predictor.predict.compile()
                    i = 0
                    epoch = 0

                test_cost_old = test_cost
                w = predictor.coef_()
                w_old = w.get_value()
                epoch += n_training_samples
            f_train(x_train[[i % n_training_samples]], y_train[[i % n_training_samples]])
            i += 1

                
        w = predictor.coef_()
        w_old = w.get_value()
        cost_voxel = f_cost(x_test,y_test)
        regr_cost["name"] = cost_voxel
        regr_coef["name"] =  w_old
            
    dd.io.save("regression_coefficients_fc.h5", regr_coef)
    dd.io.save("regression_cost_fc.h5", regr_cost)
            #print f_cost(x_test,y_test) 
            #return f_cost(x_test, y_test) # return values for testing cost function in commandline
                #print "Cost train: %d" % (f_cost(x_train, y_train[:,i:i+10]))
                #print "Cost test: %d" % (f_cost(x_test,y_test[:,i:i+10]))
