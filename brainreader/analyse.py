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

def filtervoxels(n = 1):
   
    print "Comparing layers in regions %s" % (n)
    conv5 = dd.io.load('/data/regression_cost_roi%s_conv5_4.h5' % (n) )
    fc6 = dd.io.load('/data/regression_cost_roi%s_fc6.h5' % (n) )
    fc7 = dd.io.load('/data/regression_cost_roi%s_fc7.h5' % (n) )
    fc8 = dd.io.load('/data/regression_cost_roi%s_fc8.h5' % (n) )
    conv5_i = dd.io.load('/data/regression_cost_init_roi%s_conv5_1.h5' % (n) )
    fc6_i = dd.io.load('/data/regression_cost_init_roi%s_fc6.h5' % (n) )
    fc7_i = dd.io.load('/data/regression_cost_init_roi%s_fc7.h5' % (n) )
    fc8_i = dd.io.load('/data/regression_cost_init_roi%s_fc8.h5' % (n) )
    conv5_new = OrderedDict()
    fc6_new = OrderedDict()
    fc7_new = OrderedDict()
    fc8_new = OrderedDict()
    for i in range (1,120):
        conv5_new[i] = (conv5[i][np.where(np.less(conv5[i] < conv5_i[i]))], np.less(conv5[i] < conv5_i[i])))
        fc6_new[i] = (fc6[i][np.where(np.less(fc6[i] < fc6_i[i]) )], np.where(np.less(fc6[i] < fc6_i[i])))
        fc7_new[i] = (fc7[i][np.where(np.less(fc6[i] < fc7_i[i]))], np.where(np.less(fc7[i] < fc7_i[i])))
        fc8_new[i] = (fc8[i][np.where(np.less(fc8[i] < fc8_i[i]) )], np.where(np.less(fc8[i] < fc8_i[i])))
    return conv5_new,fc6_new, fc7_new,fc8_new
   

def count_layers(region):
    region_most = OrderedDict()
    for n in range(1,7):
        voxel_layer = region[n]
        pair = OrderedDict()
        fc6 = 0
        conv5 = 0
        fc7 = 0
        fc8 = 0
        for i in range(0, len(voxel_layer)):
            if voxel_layer[i] ==  'conv5':
                conv5 += 1
                
            elif voxel_layer[i] ==  'fc6':
                fc6 += 1
            elif voxel_layer[i] ==  'fc7':
                fc7 += 1
            else :
                fc8 += 1
        print fc7
        print fc8
        print max(conv5,fc6,fc7,fc8)
        if max(conv5,fc6,fc7,fc8) == conv5:
            region_most[n] = 'conv5'
        elif  max(conv5,fc6,fc7,fc8) == fc6:
            region_most[n] = 'fc6'
        elif  max(conv5,fc6,fc7,fc8) == fc7:
            region_most[n] = 'fc7'

        elif max(conv5,fc6,fc7,fc8) == fc8:
            region_most[n] = 'fc8'
    return region_most
            
        
        


