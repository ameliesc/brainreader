import theano.tensor as tt
from plato.core import symbolic, create_shared_variable
from collections import OrderedDict
from plato.core import symbolic, create_shared_variable
from plato.interfaces.helpers import get_named_activation_function
from theano import In, function
import numpy as np


@symbolic
class Nonlinearity(object):

    def __init__(self, activation):
        """
        activation:  a name for the activation function. {'relu', 'sig', 'tanh', ...}
        """
        self.activation = get_named_activation_function(activation)

    def __call__(self, x):
        return self.activation(x)

@symbolic
class Deconv(object):

    def __init__(self, w, b, force_shared_parameters = True, border_mode = 'full', filter_flip = True):
        """
        w is the kernel, an ndarray of shape (n_output_maps, n_input_maps, w_size_y, w_size_x)
        b is the bias, an ndarray of shape (n_output_maps, )
        force_shared_parameters: Set to true if you want to make the parameters shared variables.  If False, the
            parameters will be
        :param border_mode: {'valid', 'full', 'half', int, (int1, int2)}.  Afects
            default is 'valid'.  See theano.tensor.nnet.conv2d docstring for details.
        """
        w = np.swapaxes(w,0,1)[:, :, ::-1, ::-1] # transpose filters for deconv according to zeiler
        self.w = create_shared_variable(w) if force_shared_parameters else tt.constant(w) #same as conv just with
        b = np.swapaxes(b,0,1)[:, :, ::-1, ::-1]
        self.b = create_shared_variable(b) if force_shared_parameters else tt.constant(b)
        self.border_mode = border_mode
        self.filter_flip = filter_flip
        # + self.b[ :,None,None]
        # - self.b[:, None, None]
    def __call__(self, x):
        """
        param x: A (n_samples, n_input_maps, size_y, size_x) image/feature tensor
        return: A (n_samples, n_output_maps, size_y-w_size_y+1, size_x-w_size_x+1) tensor
        """
        return tt.nnet.conv2d(input= x, filters=self.w, border_mode=self.border_mode, filter_flip=self.filter_flip)

    @property
    def parameters(self):
        return [self.w, self.b]
@symbolic
class Unpooler(object):

    def __init__(self, switch):
        self.switch = switch
        
    def __call__(self, x):
        """
        Input:
        - x: an (n_samples, n_maps, size_y, size_x) np.array for now
        Output
        - (n_samples, n_maps, size_y*ds[0], size_x*ds[1]) tensor
        """
        return x.repeat(2,axis =2).repeat(2, axis =3)*self.switch

    # @symbolic
    # def unpooler(self,input):
    #     #x = x.repeat(2,axis =2).repeat(2, axis =3)
    #     s = tt.ftensor4('s')
    #     y = tt.ftensor4('y')
    #     f = function([y, In(s, value=self.switch)], y*s)
    #     return f(input.repeat(2,axis =2).repeat(2, axis =3))
    


    
@symbolic
class DeconvNet(object):

    def __init__(self,layers):
        self.n_layers = len(layers)
        self.layers = layers

    def __call__(self,x):
        for name,layer in self.layers.iteritems():
            print '%s input shape: %s' % (name, x.ishape)
            x = layer(x)
            print '%s output shape: %s' % (name, x.ishape)
        return x
