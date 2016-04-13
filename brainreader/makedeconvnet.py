from fileman.file_getter import get_file
from scipy.io import loadmat
import copy
import pickle


def load_conv_and_deconv(save = "no", network_params_conv = None):
    if network_params_conv is None:
        filename = get_file(
            relative_name='data/vgg-19.mat',
            url='http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat',
        )
        network_params_conv = loadmat(filename)

    #pool - unpool
    #relu - relunl
    # conv - decon ; transpose learned filters (appled to recitfied map not
    # output beneath)

    
    network_params_deconv = copy.deepcopy(network_params_conv)
    print "Inverting VGG Network..."
    j = network_params_conv['layers'].shape[1] - 1
    for i in range(0, network_params_conv['layers'].shape[1]):
        layer_type = network_params_conv['layers'][0][j][0][0][1][0]
        if layer_type == 'relu':
            new_layer_type = layer_type + 'nl'
        elif layer_type == 'pool':
            new_layer_type = 'un' + layer_type
        elif layer_type == 'conv':
            new_layer_type = 'de' + layer_type
        else:
            new_layer_type = layer_type
        network_params_deconv['layers'][0][i] = network_params_conv['layers'][0][j]
        network_params_deconv['layers'][0][i][0][0][1][0] = new_layer_type
        j = j - 1
        
    if save == "yes":
        print "Writing to file ..."
        out =  open("deconvnetwork.p", "w")
        pickle.dump(network_params_deconv, out)
        out.close()
    print "Done."
    return network_params_deconv
    #for i in range(0,network_params_deconv['layers'].shape[1]):
     #   print str(network_params_deconv['layers'][0,i][0, 0][1][0]) + ' '
    #return network_params_deconv, network_params_conv
