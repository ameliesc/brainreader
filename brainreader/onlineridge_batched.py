import numpy as np
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressiontheano import LinearRegressor

def online_ridge():

    #['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2',
    #'conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1','conv4_2', 'conv4_3',  'conv4_4', 'conv5_1','conv5_2',  'conv5_3',   'conv5_4',
    layer_names = [  'fc6', 'fc7','fc8']

    regr_coef  = OrderedDict()
    regr_cost = OrderedDict()
    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
        print "Done."

        y_train = get_data(response=1, roi = 4)
        x_train = feature_map_train
        y_test = get_data(response=1, data='test', roi = 4 )  # n_samples x n_targets
        x_test = feature_map_test  # n_samples x n_feature
        batch_size = 20
        n_in = x_train.shape[1]
        n_out = batch_size
        n_training_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]
        score_report_period = 500
        n_epochs = 100
        lmbda = 0.1
        cost_voxel = np.zeros_like(y_test)
        weights_voxel = np.zeros((x_train.shape[1],y_train.shape[1]))
        j = 0
        last_inf_eta = 0
        while j < y_train.shape[1]:
            print "training batch %s" % (j / batch_size)
            if y_train.shape[1] - j < batch_size:
                n_out = y_train.shape[1] - j
            eta = last_inf_eta
            predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = 2 )
            f_train = predictor.train.compile()
            f_predict = predictor.predict.compile()
            f_cost = predictor.voxel_cost.compile()
            # Train on one sample at a time and periodically report score.
            epoch = 0
            test_cost_old = 500
            roh = 1.1
            alpha = 0.5
            w_old = None
            i = 0
            
            while i < n_training_samples*n_epochs+1:
                
                if i % score_report_period == 0:
                    out = f_predict(x_test)
                    test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0) 
                
                    print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
                    
                if np.isnan(test_cost) or np.isinf(test_cost):
                    print "Cost (%s) resetting parameters." % (test_cost)
                    i = 0
                    
                    eta = predictor.get_params()
                    eta = eta * 0.5
                    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old)
                    f_train = predictor.train.compile()
                    f_predict = predictor.predict.compile()
                    f_cost = predictor.voxel_cost.compile()
                    last_inf_eta = eta
                    print "new step size: %s" % (eta)
                    continue
            

                if i == epoch: # Adaptive Stepsize

                    out = f_predict(x_test)
                    test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0) 
                
 
                    if test_cost_old < test_cost:
                        print "Cost too high (%s)" % (test_cost)
                        eta = predictor.get_params()
                        eta = 0.5 * eta
                        predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old,cost = test_cost )
                        f_train = predictor.train.compile()
                        f_predict = predictor.predict.compile()
                        f_cost = predictor.voxel_cost.compile()
                        print "Decreasing  learning rate to %s" % (eta)
                    

                    elif np.isnan(test_cost) or np.isinf(test_cost):
                        print "Cost (%s) resetting parameters." % (test_cost)
                        eta = predictor.get_params()
                        eta = 0.5*eta
                        predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old, cost = test_cost )
                        f_train = predictor.train.compile()
                        f_predict = predictor.predict.compile()
                        f_cost = predictor.voxel_cost.compile()
                        print "Decreasing learning rate: %s" % (eta)
                        i = 0
                        

                    elif test_cost_old > test_cost or test_cost_old == test_cost:
                        eta = predictor.get_params()
                        eta_old = eta
                        eta = 1.1 * eta
                        
                        predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = w_old)
                        
                        f_train = predictor.train.compile()
                        f_predict = predictor.predict.compile()
                        f_cost = predictor.voxel_cost.compile()
                        print "Increasing learning rate from %s to  %s" % (eta_old, eta)


                    if abs(test_cost_old - test_cost) < eta * 0.001 and epoch > 8 * n_training_samples : # stopping condition
                        print "No learning, move to next batch"
                        i = n_training_samples*n_epochs+1

                    elif abs(test_cost_old - test_cost) < eta * 0.001 and epoch >= 5 * n_training_samples:
                        print "no improvement, randomly initialising the weights"
                        predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, eta = eta, w = np.array([0]) )  # random initialisation, might be stuck in local minimum
                        f_train = predictor.train.compile()
                        f_predict = predictor.predict.compile()
                        f_cost = predictor.voxel_cost.compile()

                    
                    
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
            
        dd.io.save("regression_coefficients_roi3_%s.h5" % (name), weights_voxel)
        dd.io.save("regression_cost_roi3_%s.h5" % (name), cost_voxel)

