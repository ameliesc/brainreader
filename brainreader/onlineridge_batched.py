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
        n_epochs = 10
        lmbda = 0.01
        cots_voxel = np.zeros_like(y_test)
        weights_voxel = np.zeros((x_train.shape[1],y_train.shape[1]))
        j = 0
        while j < y_train.shape[1]:
            print "training batch "
            predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = 0.01 )
            f_train = predictor.train.compile()
            f_predict = predictor.predict.compile()
            set_params = predictor.set_params.compile()
            f_cost = predictor.voxel_cost.compile()
            # Train on one sample at a time and periodically report score.
            epoch = 0
            test_cost_old = 0
            roh = 1.1
            alpha = 0.5
            w_old = None
        
            i = 0
            while i < n_training_samples*n_epochs+1:
                out = f_predict(x_test)
                test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0) 
                #skip NaN reslts
                if test_cost_old < test_cost:
                    test_cost = test_cost
                elif test_cost_old > test_cost or test_cost_old == test_cost:
                    test_cost = test_cost
                else:
                    print "Cost nan or inf resetting parameters."
                    predictor.set_params(alpha=alpha, w = w_old)
                    i = 0
                    epoch = 0
                    eta = predictor.get_params()
                    eta = eta.get_value()
                    print "new step size: %s" % (eta)
                    
                if i % score_report_period == 0:
                    print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)

                if i == epoch: # Adaptive Stepsize
 
                    if test_cost_old < test_cost:

                        print "Cost too high resetting parameters."
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        predictor.set_params(alpha=alpha, roh = 1, eta = eta, w = w_old)
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        print "new step size: %s" % (eta)
                        #set_params(alpha = alpha)

                    elif test_cost_old > test_cost or test_cost_old == test_cost:
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        predictor.set_params(roh = roh, alpha = 1, eta = eta)
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        print "Increasing learning rate to : %s" % (eta)
                        
                    else:
                        print "Cost nan or inf resetting parameters."
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        predictor.set_params(alpha=alpha, roh = 1, eta = eta, w = w_old)
                        eta = predictor.get_params()
                        eta = eta.get_value()
                        print "new step size: %s" % (eta)

                        i = 0
                        epoch = 0

                    test_cost_old = test_cost
                    w = predictor.coef_()
                    w_old = w.get_value()
                    epoch += n_training_samples
                    f_train(x_train[[i % n_training_samples]], y_train[i % n_training_samples, j: j+batch_size])
                    i += 1
            cost_batch = f_cost(x_test, y_test[:,j:j+batch_size])
            cost_voxel[:,j:j+batch_size] =cost_batch
            w = predictor.coef_()
            w_old = w.get_value()
            weights_voxel[:,j:j+batch_size] = w_old
            j = j + batch_size
            
        regr_cost[name] = cost_voxel
        regr_coef[name] = weights_voxel
            
    dd.io.save("regression_coefficients_roi3_fc.h5", regr_coef)
    dd.io.save("regression_cost_roi3_fc.h5", regr_cost)
            #print f_cost(x_test,y_test) 
            #return f_cost(x_test, y_test) # return values for testing cost function in commandline
                #print "Cost train: %d" % (f_cost(x_train, y_train[:,i:i+10]))
                #print "Cost test: %d" % (f_cost(x_test,y_test[:,i:i+10]))
