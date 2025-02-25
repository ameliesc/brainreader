import numpy as np
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressionridgev2 import LinearRegressor

def online_ridge():

    # ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2','conv3_1', 'conv3_2',  'conv3_3',  'conv3_4', 'conv4_1','conv4_2', 'conv4_3',  'conv4_4',
    layer_names = [ 'conv5_1','conv5_2',  'conv5_3', 'conv5_4', 'conv5_4', 'fc6', 'fc7','fc8']

    regr_coef  = OrderedDict()
    regr_cost = OrderedDict()
    for name in layer_names:

        print "load featuremap for training.."
        feature_map_train = dd.io.load("featuremap_train_%s.h5" % (name))
        print "load featuremap for testing.."
        feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
        print "Done."

        y_train = get_data(response=1, roi = 6)
    
        x_train = np.nan_to_num((feature_map_train-np.mean(feature_map_train, axis=1)[:, None])/np.std(feature_map_train, axis=1)[:, None])
        y_test = get_data(response=1, data='test', roi = 6 )  # n_samples x n_targets
        x_old = feature_map_test
        x_test = np.nan_to_num((feature_map_test-np.mean(feature_map_test, axis=1)[:, None])/np.std(feature_map_test, axis=1)[:, None])
        
 # n_samples x n_feature
        batch_size = 1
        n_in = x_train.shape[1]
        n_out = batch_size
        n_training_samples = x_train.shape[0]
        n_test_samples = x_test.shape[0]
        score_report_period = 1000
        n_epochs = 1000
        lmbda = 0.01
        cost_voxel = np.zeros_like(y_test)
        weights_voxel = np.zeros((x_train.shape[1],y_train.shape[1]))
        j = 0
        
        while j < y_train.shape[1]:

            print "training batch %s" % (j / batch_size)
            epoch = 0
            test_cost = 0
            test_cost_old = 0
            cost_min = 100
            if y_train.shape[1] - j < batch_size:
                n_out = y_train.shape[1] - j
            
            predictor = LinearRegressor(n_in, n_out, lmbda = lmbda)
            f_train = predictor.train.compile()
            f_predict = predictor.predict.compile()
            f_cost = predictor.voxel_cost.compile()
            i = 0
            while i < n_training_samples*n_epochs+1:
                if i % score_report_period == 0:
                    out = f_predict(x_test)
                    test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0)
                    cost_min = min(cost_min,test_cost)

                    print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)

                #if i == epoch:
                    #if  abs(test_cost_old - test_cost) < 1e-9 and epoch > 6 *  n_training_samples: #stopping condition
                        # print "im loop"
                        # print test_cost
                        # print cost_min
                        # if test_cost != cost_min:
                            
                        #     print "stuck in local minimum, resetting weights randomly"
                        #     predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, w =0 )
                        #     f_train = predictor.train.compile()
                        #     f_predict = predictor.predict.compile()
                        #     f_cost = predictor.voxel_cost.compile()
                        #     epoch = 0
                        #     i = 0
                        #     test_cost = 0
                        #     test_cost_old = 0
                    
                    #    else:
                        #print "No learning, move to next batch"
                        #break
                    #test_cost_old = test_cost    
                    #epoch += n_training_samples

                f_train(x_train[[i % n_training_samples]], y_train[i % n_training_samples, j: j+batch_size])
                i += 1

            cost_batch = f_cost(x_test, y_test[:,j:j+batch_size])
            cost_voxel[:,j:j+batch_size] =cost_batch
            w = predictor.parameters
            weights_voxel[:,j:j+batch_size] = w[0].get_value()
            j = j + batch_size
        regr_cost[name] = cost_voxel
        regr_coef[name] = weights_voxel
        dd.io.save("regression_coefficients_roi6_%s.h5" % (name), weights_voxel)
        dd.io.save("regression_cost_roi6_%s.h5" % (name), cost_voxel)
    dd.io.save("regression_all_coef_roi6.h5", regr_coef)
    dd.io.save("regression_all_cost_roi6.h5", regr_cost)

