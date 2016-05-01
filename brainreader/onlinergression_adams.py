from plato.core import create_shared_variable, symbolic, add_update
import theano.tensor as tt
import numpy as np
import theano

class LinearRegressor:

    def __init__(self, n_in, n_out, lmbda, alpha = 0.001):
        self.w = theano.shared(np.zeros((n_in, n_out)))
        self.lmbda = lmbda
        self.eta = 1e-8
        self.alpha = alpha
        self.m_v1 = theano.shared(np.zeros((n_in,n_out)))
        self.v_v2 = theano.shared(np.zeros((n_in,n_out)))
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 0
        
        
    @symbolic
    def train(self, x, targ):# x: (n_samples, n_in), targ: (n_samples, n_out)       
        y = self.predict(x)
        self.t += 1
        cost  =  (((targ - y)**2) + (self.lmbda * (self.w  ** 2).sum(axis = 0))).sum(axis=1).mean(axis=0)
        add_update(self.m_v1, self.beta1 * self.m_v1 + (1-self.beta1) * cost)
        add_update(self.v_v2, self.beta2 * self.v_v2 + (1-self.beta2) * (cost ** 2))
        m_bias = self.m_v1/(1 - self.beta1 ** self.t)
        v_bias = self.v_v2/(1 - self.beta2 ** self.t)
        add_update(self.w, self.w - self.alpha * m_bias/(tt.sqrt(self.v_v2) + self.eta)) # update with
        
    @symbolic
    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)

    @symbolic
    def voxel_cost(self, x,y_true):
        y = self.predict(x) 
        return ((y_true - y)**2).sum(axis=0)

    def coef_(self):
        return self.w

    def get_params(self):
        return self.alpha
