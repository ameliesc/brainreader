from plato.core import create_shared_variable, symbolic, add_update
import theano.tensor as tt
import numpy as np

class LinearRegressor:

    def __init__(self, n_in, n_out, lmbda, eta = 0.01):
        self.w = create_shared_variable(np.zeros((n_in, n_out)))
        self.eta = eta
        self.lmbda = lmbda

    @symbolic
    def train(self, x, targ):  # x: (n_samples, n_in), targ: (n_samples, n_out)
        y = self.predict(x)
        cost  =  (((targ - y)**2) + (self.lmbda * (self.w  ** 2)).sum(axis = 0)).sum(axis=1).mean(axis=0)
        add_update(self.w, self.w - self.eta*tt.grad(cost=cost, wrt=self.w))
    

    @symbolic
    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)

    @symbolic
    def cost(self, x_test,y_true):
        y = self.predict(x_test)
        return (((y_true - y)**2) + (self.lmbda * (self.w  ** 2)).sum(axis = 0)).sum(axis=1).mean(axis=0) #returns inf?

    def coef_(self):
        return self.w().get_value()

