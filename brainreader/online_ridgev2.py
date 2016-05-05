import numpy as np
from data_preprocessing import get_data
from collections import OrderedDict
import deepdish as dd
from regressionridgev2 import LinearRegressor






def online_ridge(region, mini_batch_size = 100, batch_size = 10, method = "Adam", stepsize = 0.000001, name = 'fc6', lmbda = 0.01, epochs = 15):

    regr_coef  = OrderedDict()
    regr_cost = OrderedDict()


    print "load featuremap for training.."
    feature_map_batch1= dd.io.load("featuremaps_train_%s.h5" % (name))

    print "load featuremap for testing.."
    feature_map_test = dd.io.load("featuremaps_test_%s.h5" % (name))
    print "Done."
    roi = region
    y_train = get_data(response=1, roi = roi)
    
    x_train = np.nan_to_num((feature_map_train-np.mean(feature_map_train, axis=1)[:, None])/np.std(feature_map_train, axis=1)[:, None])
    y_test = get_data(response=1, data='test', roi = roi )  # n_samples x n_targets
    x_old = feature_map_test
    x_test = np.nan_to_num((feature_map_test-np.mean(feature_map_test, axis=1)[:, None])/np.std(feature_map_test, axis=1)[:, None])
        
    # n_samples x n_feature
    batch_size = batch_size
    sample_batch_size = mini_batch_size
    n_in = x_train.shape[1]
    n_out = batch_size
    n_training_samples = x_train.shape[0]
    n_test_samples = x_test.shape[0]
    score_report_period = 350
    n_epochs = epochs
    lmbda = lmbda
    cost_voxel = np.zeros_like(y_test)
    weights_voxel = np.zeros((x_train.shape[1],y_train.shape[1]))
    j = 0
    predictor = LinearRegressor(n_in, n_out, lmbda = lmbda, method = method, stepsize = stepsize)
    f_train = predictor.train.compile()
    f_predict = predictor.predict.compile()
    f_cost = predictor.voxel_cost.compile()
        
    while j < y_train.shape[1]:  # batches for targest
        f_train = predictor.train.compile()
        f_predict = predictor.predict.compile()
        f_cost = predictor.voxel_cost.compile()
        i =  0
        if y_train.shape[1] - j < batch_size: #discard last batches 
            n_out = y_train.shape[1] - j
            break
            
           
        while i < n_training_samples*n_epochs+1: # training sample batched
            

            if i % score_report_period == 0:
                out = f_predict(x_test)
                test_cost = ((y_test[:,j : j+ batch_size] - out)**2).sum(axis = 1).mean(axis=0)
                print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)

            f_train(x_train[i: i+sample_batch_size,:], y_train[i: i+sample_batch_size, j: j+batch_size])
            i += sample_batch_size
         
        cost_batch = f_cost(x_test, y_test[:,j:j+batch_size])
        cost_voxel[:,j:j+batch_size] =cost_batch
        w = predictor.parameters
        weights_voxel[:,j:j+batch_size] = w[0].get_value()
        j = j + batch_size
    regr_predictions = f_predict(x_test)
    dd.io.save("regression_coefficients2_roi%s_%s.h5" % (roi,name), regr_predictions)
    dd.io.save("regression_coefficients2_roi%s_%s.h5" % (roi,name), weights_voxel)
    dd.io.save("regression_cost2_roi%s_%s.h5" % (roi,name), cost_voxel)

if __name__ == '__main__':
    for name in ['fc8','fc7','fc6']:
        print name + " regression ..."
        for i in xrange(1,8):
            online_ridge(region = i, stepsize=0.00000001, epochs = 15, name=name)
        print "Done."

    for name in ['conv5_4','conv5_3','conv5_2','conv5_1']:
        print name + " regression ..."
        for i in xrange(1,8):
            online_ridge(region = i, stepsize=0.00000001, epochs = 15, name=name)
        print "Done."
   
            

