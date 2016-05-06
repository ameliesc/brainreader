import deepdish as dd
import numpy as np
from collections import OrderedDict

def make_touple(array, name):
    for i in range(0, array.shape[0]):
        array[i] = (array[i],name)
    return array

def arrange():
    regions_coef = OrderedDict()
    for n in range(1,8):
        coef = OrderedDict()
        for name in ['conv5_4','fc6', 'fc7', 'fc8']:
            coef[name] = dd.io.load('regression_coefficients_roi%s_%s.h5' % (n, name))
        regions_coef[n] = coef
        dd.io.save('region_coef.h5', regions_coef)

def filtervoxels(layer_name, n = 1):
   
    print "Filtering voxels in layer %s in regions %s" % (layer_name,n)
 
    layer = dd.io.load('/data/regression_cost_roi%s_%s.h5' % (n,layer_name))
   
    layer_init = dd.io.load('/data/regression_cost_init_roi%s_%s.h5' % (n,layer_name))
    layer_new = OrderedDict()
    for i in range (1,120): #   conv5_new[i] = (conv5[i][np.where(np.less(conv5[i] < conv5_i[i]))], np.where(np.less(conv5[i] < conv5_i[i])))
        layer_new[i] =  (layer[i][np.where(np.less(layer[i] , layer_init[i]) )],  np.where(np.less(layer[i] ,layer_init[i])))
    return layer_new

def filtercost(dic):
    cost = dic[1][0]
    index = dic[1][1][0]
    index_1 = np.where(cost < 10)
    index = index[index_1]
    return index
   

def mincost():
    layers = OrderedDict()
    for layer in ['fc6', 'fc7', 'fc8']:
        dic = filtervoxels(layer, n =roi)
        index = filtercost(dic)
        layers[layer] = min(dic[1][0][index])
    return layers
        
def avgcost():  
    layers = OrderedDict()
    for layer in ['fc6', 'fc7', 'fc8']:
        dic = filtervoxels(layer, n =roi)
        index = filtercost(dic)
        layers[layer] = sum(dic[1][0][index])/len(dic[1][0][index])
    return layers
        


