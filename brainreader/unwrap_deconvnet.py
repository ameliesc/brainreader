from collections import OrderedDict
from deconvnets import  Nonlinearity, Unpooler, Deconv, DeconvNet
from general.should_be_builtins import bad_value
from makedeconvnet import load_conv_and_deconv

def get_deconv( switches,network_params = None,from_layer = None, force_shared_parameters=True):

    if network_params is None:
        deconv_network = load_conv_and_deconv()


    def struct_to_layer(struct, switches = None):
        for i in xrange(1,network_params['layers'].shape[1]): # first layer is softmax which is skipped
            layer_type = struct[1][0]
            layer_name = str(struct[0][0])
            if layer_type == 'deco':
                w_orig = struct[2][0, 0] 
                w = w_orig.T.swapaxes(2, 3)
                b = struct[2][0, 1][:, 0]
                padding = 'full' if layer_name.startswith('fc') else 1 if layer_name.startswith('conv') else bad_value(layer_name)
                layer = Deconv(w, b, force_shared_parameters=force_shared_parameters, border_mode=padding, filter_flip=True)
            elif layer_type == 'relu':
                layer = Nonlinearity(layer_type)
            elif layer_type == 'unpo':
                switch = switches[layer_name+'_switch']
                layer = Unpooler(switch)
            else:
                raise Exception(
                "Don't know about this '%s' layer type." % layer_type)
        return layer
    print "Loading DeconvNet..."
    network_layers = OrderedDict()

    for i in range(1,network_params['layers'].shape[1]):
        layer_name = str(network_params['layers'][0,i][0,0][0][0])
        if from_layer is not None:
            if from_layer != layer_name:
                continue
            else: 
                from_layer = None
        layer = struct_to_layer(network_params['layers'][0,i][0,0],switches)
        network_layers[layer_name] = layer
    print 'Done.'
    return DeconvNet(network_layers)
