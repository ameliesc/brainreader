from plato.core import create_shared_variable, symbolic, add_update
import theano.tensor as tt
import theano as t
import numpy as np

class LinearRegressor:

    def __init__(self, n_in, n_out, lmbda, eta= 0.01, w = None):
        if w is not None:
            self.w = t.shared(w)
        else:
            self.w = t.shared(np.zeros((n_in, n_out)))
        self.lmbda = lmbda
        self.cost = None
        self.eta = t.shared(eta)

    @symbolic
    def set_params(self,roh=1, alpha =1, eta = None, w = None):
        if eta is not None:
            eta_new = roh * eta *alpha
            self.eta.set_value(eta_new)
        if w is not None:
            self.w.set_value(w)
        
    @symbolic
    def train(self, x, targ):# x: (n_samples, n_in), targ: (n_samples, n_out)       
        y = self.predict(x)
        if self.cost == None:
            delta_w_old = 0
            beta = 0
        else:
            beta = 0.95
            delta_w_old = self.w - self.eta*tt.grad(cost=self.cost, wrt=self.w) # old gradient
        self.cost  =  (((targ - y)**2) + (self.lmbda * (self.w  ** 2).sum(axis = 0))).sum(axis=1).mean(axis=0)
        add_update(self.w, beta * delta_w_old + (1-beta)*(- self.eta*tt.grad(cost=self.cost, wrt=self.w))) # update with momentum

    @symbolic
    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)

    @symbolic
    def voxel_cost(self, x,y_true):
        y = self.predict(x) 
        return ((y_true - y)**2).sum(axis=0)

    def get_params(self):
        return self.eta

    def coef_(self):
        return self.w

