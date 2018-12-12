"""Trains a Bayesian LCS with a GA on the Waterhouse (1996) test function.
"""

import sys
from numpy import double, array, ones, empty, arange, empty, hstack, \
     sqrt, exp, sort, sum, inf, power, dot, linspace, sin, pi
from numpy.random import random, randn, binomial, uniform, normal
import Gnuplot

from ga import GeneticAlgorithm_TS
from cls import RBF1DClStore, RBF1DIndv, \
     SoftIntervalClStore, SoftInterval1DIndv
from mcmc import SampleModelPosterior
from experiments import read_data, write_data, write_raw_data, plot_cls, \
     GA_experiment, MCMC_experiment

waterhouse_data_file = "exp2_waterhouse.data"
waterhouse_data_raw_file = "exp2_waterhouse_raw.data"
waterhouse_data_points = 200
own_data_file = "exp2_own.data"
own_data_raw_file = "exp2_own_raw.data"
own_data_points = 300
noise_data_file = "exp2_noise.data"
noise_data_raw_file = "exp2_noise_raw.data"
noise_data_points = 200
sinus_data_file = "exp2_sinus.data"
sinus_data_raw_file = "exp2_sinus_raw.data"
sinus_data_points = 300

def write_waterhouse_data():
    """Generates the data set and writes it to the data_file.
    """
    # generate the data x, y
    #var = 0.44
    var = 0.20
    #var = 0.05
    x = sort(random(waterhouse_data_points) * 4.0)
    y = 4.26 * (exp(-x) - 4 * exp(-2 * x) + 3 * exp(-3 * x)) \
        + sqrt(var) * randn(waterhouse_data_points)
    # write the data
    write_data(x, y, waterhouse_data_file)

def write_waterhouse_raw_data():
    """Writes the raw data without noise.
    """
    x = linspace(0, 4, 1000)
    y = 4.26 * (exp(-x) - 4 * exp(-2 * x) + 3 * exp(-3 * x))
    write_data(x, y, waterhouse_data_raw_file)

def read_waterhouse_data():
    return read_data(waterhouse_data_file)

def own_f(x):
    """Returns f(x) for given x.
    """
    # functions are
    # f1(x) = 0.05 + 0.5 x
    # f2(x) = 2 - 4 x
    # f3(x) = -1.5 + 2.5 x
    fns = array([[0.05, 0.5], [2.0, -4.0], [-1.5, 2.5]], double)
    # gaussian basis functions are given by (mu, var, weight):
    # (0.2, 0.05), (0.5, 0.01), (0.8, 0.05)
    gbfs = array([[0.2, 0.05, 0.5], [0.5, 0.01, 1.0], [0.8, 0.05, 0.4]], double)
    # plain function values
    fx = fns[:,0] + x * fns[:,1]
    #print "%f\t%f\t%f\t%f" % (x, fx[0], fx[1], fx[2])
    # mixing weights
    mx = gbfs[:,2] * exp(-0.5 / gbfs[:,1] * power(x - gbfs[:,0], 2.0))
    mx /= sum(mx)
    #print "%f\t%f\t%f\t%f" % (x, mx[0], mx[1], mx[2])    
    # return mixed function
    return dot(fx, mx)

def write_own_data():
    """Generates 'artificial' dataset and writes it to file.
    """
    noise = 0.1
    x = uniform(size = own_data_points)
    y = array([own_f(x_n) for x_n in x], double) + \
        normal(size = own_data_points) * noise
    write_data(x, y, own_data_file)

def write_own_raw_data():
    """Writes raw classifier and function to file.
    """
    x = linspace(0, 1.0, 1000)
    y = array([own_f(x_n) for x_n in x], double)
    W = array([[0.05, 0.5], [2.0, -4.0], [-1.5, 2.5]], double)
    X = hstack((ones(len(x), double).reshape(len(x), 1),
               x.reshape(len(x), 1)))
    Y = dot(X, W.T)
    write_raw_data(x, hstack([y.reshape(len(x), 1), Y]), own_data_raw_file)

def read_own_data():
    return read_data(own_data_file)

def noise_f(x):
    """function with different noise levels.
    """
    if x > 0:
        return -1.0 + 2.0 * x
    else:
        return -1.0 - 2.0 * x

def write_noise_data():
    """Generates function with different leven of noise in different
    areas of the function.
    """
    l_noise, u_noise = 0.6, 0.1
    x = uniform(-1.0, 1.0, size = noise_data_points)
    y = array([noise_f(xn) + \
               (normal(0.0, l_noise) if xn < 0 else normal(0.0, u_noise)) \
               for xn in x], double)
    write_data(x, y, noise_data_file)

def write_noise_raw_data():
    """Writes the basic function.
    """
    x = linspace(-1, 1, 1000)
    y = array([noise_f(x_n) for x_n in x], double)
    write_data(x, y, noise_data_raw_file)

def read_noise_data():
    return read_data(noise_data_file)

def write_sinus_data():
    """Generates sinusoid data with some noise.
    """
    x = uniform(-1.0, 1.0, size = sinus_data_points)
    y = sin(2 * pi * x) + normal(0.0, 0.15, size = sinus_data_points)
    write_data(x, y, sinus_data_file)

def write_sinus_raw_data():
    """Generate sinusoid data without noise.
    """
    x = linspace(-1.0, 1.0, 1000)
    y = sin(2 * pi * x)
    write_data(x, y, sinus_data_raw_file)

def read_sinus_data():
    return read_data(sinus_data_file)

def exp2a():
    """Running GA on waterhouse data.
    """
    X, Y = read_waterhouse_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = RBF1DClStore(0.0, 4.0)
    # run experiment with over 100 epochs with 20 individuals in the pop.
    GA_experiment(X, Y, Xf, 250,
                  [1 + binomial(4, 0.5) for p in xrange(20)],
                  cl_store, RBF1DIndv,
                  'exp2a_fitness.data', 'exp2a_cls.data')

def exp2b():
    """Running MCMC on waterhouse data.
    """
    X, Y = read_waterhouse_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = RBF1DClStore(0.0, 4.0)
    MCMC_experiment(X, Y, Xf, 500, 10, 0.25,
                    1 + binomial(4, 0.5),
                    cl_store,
                    'exp2b_varbound.data', 'exp2b_cls.data')

def exp2c():
    """Running GA on own data.
    """
    X, Y = read_own_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = RBF1DClStore(0.0, 1.0)
    # run experiment with over 100 epochs with 20 individuals in the pop.
    GA_experiment(X, Y, Xf, 250,
                  [1 + binomial(8, 0.5) for p in xrange(20)],
                  cl_store, RBF1DIndv,
                  'exp2c_fitness.data', 'exp2c_cls.data')
    
def exp2d():
    """Running MCMC on own data.
    """
    X, Y = read_own_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = RBF1DClStore(0.0, 1.0)
    MCMC_experiment(X, Y, Xf, 500, 10, 0.25,
                    1 + binomial(8, 0.5),
                    cl_store,
                    'exp2d_varbound.data', 'exp2d_cls.data')

def exp2e():
    """Running GA on noisy data, using soft interval classifiers.
    """
    X, Y = read_noise_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = SoftIntervalClStore(-1.0, 1.0)
    # run experiment with over 100 epochs with 20 individuals in the pop.
    GA_experiment(X, Y, Xf, 250,
                  [1 + binomial(8, 0.5) for p in xrange(20)],
                  cl_store, SoftInterval1DIndv,
                  'exp2e_fitness.data', 'exp2e_cls.data')
    
def exp2f():
    """Running MCMC on noisy data, using soft interval classifiers.
    """
    X, Y = read_noise_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = SoftIntervalClStore(-1.0, 1.0)
    MCMC_experiment(X, Y, Xf, 500, 10, 0.25,
                    1 + binomial(8, 0.5),
                    cl_store,
                    'exp2f_varbound.data', 'exp2f_cls.data')

def exp2g():
    """Running GA on sinusoid data, using soft interval classifiers.
    """
    X, Y = read_sinus_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = SoftIntervalClStore(-1.0, 1.0)
    # run experiment with over 100 epochs with 20 individuals in the pop.
    GA_experiment(X, Y, Xf, 250,
                  [1 + binomial(8, 0.5) for p in xrange(20)],
                  cl_store, SoftInterval1DIndv,
                  'exp2g_fitness.data', 'exp2g_cls.data')
    
def exp2h():
    """Running MCMC on sinusoid data, using soft interval classifiers.
    """
    X, Y = read_sinus_data()
    N = X.shape[0]
    Xf = ones(N, double).reshape(N, 1)
    cl_store = SoftIntervalClStore(-1.0, 1.0)
    MCMC_experiment(X, Y, Xf, 500, 10, 0.25,
                    1 + binomial(8, 0.5),
                    cl_store,
                    'exp2h_varbound.data', 'exp2h_cls.data')


# run experiments from arguments
if __name__ == '__main__':
    exp_modes = {'gen1': lambda: write_waterhouse_data(),
                 'gen2': lambda: write_own_data(),
                 'gen3': lambda: write_noise_data(),
                 'gen4': lambda: write_sinus_data(),
		 'raw1': lambda: write_waterhouse_raw_data(),
		 'raw2': lambda: write_own_raw_data(),
		 'raw3': lambda: write_noise_raw_data(),
                 'raw4': lambda: write_sinus_raw_data(),
                 'a': lambda: exp2a(),
                 'b': lambda: exp2b(),
                 'c': lambda: exp2c(),
                 'd': lambda: exp2d(),
                 'e': lambda: exp2e(),
                 'f': lambda: exp2f(),
                 'g': lambda: exp2g(),
                 'h': lambda: exp2h()}
    for argv in sys.argv[1:]:
        if not exp_modes.has_key(argv):
            print "--- Unkown experiment: %s" % argv
        else:
            print "--- Running '%s'" % argv
            exp_modes[argv]()
