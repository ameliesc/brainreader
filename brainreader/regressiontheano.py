from plato.core import create_shared_variable, symbolic, add_update
import theano.tensor as tt
import numpy as np

# Set up parameters
n_in = 80000
n_out = 1300
n_training_samples = 1750
n_test_samples = 1750
n_epochs = 2
noise = 0.1
random_seed = 1234
score_report_period = 100

# Create a regression dataset
rng = np.random.RandomState(random_seed)
w_true = rng.randn(n_in, n_out)  # (n_in, n_out)
training_data = rng.randn(n_training_samples, n_in)  # (n_training_samples, n_in)
training_target = training_data.dot(w_true) + noise*rng.randn(n_training_samples, n_out)  # (n_training_samples, n_out)
test_data = rng.randn(n_test_samples, n_in)  # (n_test_samples, n_in)
test_target = test_data.dot(w_true) + noise*rng.randn(n_test_samples, n_out)  # (n_test_samples, n_out)

# Create a linear regressor
class LinearRegressor:

    def __init__(self, n_in, n_out, eta = 0.01):
        self.w = create_shared_variable(np.zeros((n_in, n_out)))
        self.eta = eta

    @symbolic
    def train(self, x, targ):  # x: (n_samples, n_in), targ: (n_samples, n_out)
        y = self.predict(x)
        cost = ((targ - y)**2).sum(axis=1).mean(axis=0)
        add_update(self.w, self.w - self.eta*tt.grad(cost=cost, wrt=self.w))

    @symbolic
    def predict(self, x):  # x: (n_samples, n_in)
        return x.dot(self.w)

# Setup the predictor and compile functions
predictor = LinearRegressor(n_in, n_out)
f_train = predictor.train.compile()
f_predict = predictor.predict.compile()

# Train on one sample at a time and periodically report score.
for i in xrange(n_training_samples*n_epochs+1):
    if i % score_report_period == 0:
        out = f_predict(test_data)
        test_cost = ((test_target-out)**2).sum(axis=1).mean(axis=0)
        print 'Test-Cost at epoch %s: %s' % (float(i)/n_training_samples, test_cost)
    f_train(training_data[[i % n_training_samples]], training_target[[i % n_training_samples]])
