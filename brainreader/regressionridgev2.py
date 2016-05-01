from plato.core import create_shared_variable, symbolic, add_update
import theano.tensor as tt
import theano 
import numpy as np
from plato.tools.optimization.optimizers import RMSProp, AdaGrad, Adam

class LinearRegressor:

    def __init__(self, n_in, n_out, lmbda, eta= 0.01, w = None):
        self.optimizer = RMSProp(learning_rate = 0.001)
        if w is None:
            self.w = theano.shared(np.zeros((n_in, n_out)))
        else:
            self.w = theano.shared(np.random.rand(n_in,n_out))
        self.lmbda = lmbda
        self.eta = eta

    @symbolic
    def train(self, x, targ):# x: (n_samples, n_in), targ: (n_samples, n_out)
        y = self.predict(x)
        #cost  = tt.mean(tt.sum((y-targ)**2, axis = 1), axis = 0)
        #add_update(self.w, self.w - self.eta*tt.grad(cost=cost, wrt=self.w))
        cost  = tt.mean(tt.sum((y-targ)**2 + tt.sum(self.lmbda * (self.w  ** 2),axis = 0), axis = 1), axis = 0)
        self.optimizer(cost = cost, parameters = self.parameters) 
        
    @symbolic
    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)

    @symbolic
    def voxel_cost(self, x,y_true):
        y = self.predict(x) 
        return ((y_true - y)**2).sum(axis=0)

    @property
    def parameters(self):
        return [self.w]

