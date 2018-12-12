import sys
from numpy import double, array, zeros, ones, arange, empty, \
     sum, power, dot, exp, sqrt
from numpy.random import uniform, normal, randint
import Gnuplot

from cls import RBF1DCl
from cl import Gating

# functions are
# f1(x) = 0.05 + 0.5 x
# f2(x) = 2 - 4 x
# f3(x) = -1.5 + 2.5 x
fns = array([[0.05, 0.5], [2.0, -4.0], [-1.5, 2.5]], double)
# gaussian basis functions are given by (mu, var, weight):
# (0.2, 0.05), (0.5, 0.01), (0.8, 0.05)
gbfs = array([[0.2, 0.05, 0.5], [0.5, 0.01, 1.0], [0.8, 0.05, 0.4]], double)
# name of data file
data_file = "exp1.data"


def f(x):
    """Returns f(x) for given x.
    """
    # plain function values
    fx = fns[:,0] + x * fns[:,1]
    #print "%f\t%f\t%f\t%f" % (x, fx[0], fx[1], fx[2])
    # mixing weights
    mx = gbfs[:,2] * exp(-0.5 / gbfs[:,1] * power(x - gbfs[:,0], 2.0))
    mx /= sum(mx)
    #print "%f\t%f\t%f\t%f" % (x, mx[0], mx[1], mx[2])    
    # return mixed function
    return dot(fx, mx)


def write_samples(n, noise = 0.0):
    """Writes n samples of f(x) in the format x\tf(x) to "exp1.data".
    If noise is given, then gaussian noise N(0, noise) is added to the return
    values. f is sampled over [0, 1].
    """
    o = open(data_file, 'w')
    for i in xrange(n):
        x = uniform()
        print >>o, "%f\t%f" % (x, f(x) + normal() * noise)


def read_data():
    """Returns the input matrix and the output vector. The input matrix is
    automatically augmented by a bias term (the first column), followed by
    a column of x-values.
    """
    # read x, fx from data file
    x, y = [], []
    r = open(data_file, 'r')
    for l in r.readlines():
        if l[-1] == '\n':
            l = l[:-1]
        l = l.strip()
        if l != '':
            xn, yn = map(float, l.split("\t"))
            x.append(xn)
            y.append(yn)
    # return augmented matrices
    return (array([[1.0] * len(x), x], double).T,
            array(y, double).reshape(len(y), 1))


def independent_cls():
    """Returns a list of independently trained classifiers, created accoring
    to gbfs.
    """
    X, Y = read_data()
    cls = []
    for k in xrange(gbfs.shape[0]):
        mu, var = gbfs[k, 0:2]
        cl = RBF1DCl(X, 1, array([mu, var], double))
        id, L = cl.update(X, Y)
        if id < -1:
            print "Exceeded iteration limit when training classifier"
        cls.append(cl)
    return cls


def plot_cls(X, Y, gate, filename=""):
    """Plots the data, the classifier prediction, and the mixed prediction.
    If a filename is given, then the prediction data is also written to a file
    with the given filename.
    """
    cls = gate.cls
    N, K = X.shape[0], len(cls)
    x = X[:,1]
    y = Y.reshape(N)
    Xf = ones(N, double).reshape(N, 1)
    plot_data = [ Gnuplot.Data(x.tolist(), y.tolist(), title="f(x)") ,]
    
    # get classifier predictions
    N = 100
    x = arange(0, N) / float(N)
    Pred = empty((N, K+3), double)
    xf = ones(1, double)
    for k in xrange(K):
        for n in xrange(N):
            Pred[n, k] = cls[k].pred(array([1, x[n]], double))
        plot_data.append(
            Gnuplot.Data(x.tolist(), Pred[:,k].tolist(),
                         title="f%i(x)" % (k + 1),
                         with="lines"))
    # get mixed prediction with variance
    for n in xrange(N):
        mean, var = gate.pred_var(array([1, x[n]], double), xf)
        Pred[n, K] = mean[0]
        Pred[n, K+1], Pred[n, K+2] = mean[0] - sqrt(var[0]), \
                                     mean[0] + sqrt(var[0])
    plot_data.append(
        Gnuplot.Data(x.tolist(), Pred[:,K].tolist(),
                     title="pred", with="lines"))
    plot_data.append(
        Gnuplot.Data(x.tolist(), Pred[:,K+1].tolist(),
                     title="pred-", with="lines"))
    plot_data.append(
        Gnuplot.Data(x.tolist(), Pred[:,K+2].tolist(),
                     title="pred+", with="lines"))
    g = Gnuplot.Gnuplot()
    g.plot(*plot_data)
    # write to file, if requested
    if filename != "":
        data_str = '\t'.join(["%0.6f"] * (K + 4))
        f = open(filename, 'w')
        for n in xrange(N):
            print >>f, data_str % tuple([x[n]] + list(Pred[n,:]))
    return g


def exp1a():
    """Experiment 1A.
    Train gating network based on independent classifiers,
    and visualise trained function.
    """
    X, y = read_data()
    N = X.shape[0]
    x, Xf = X[:,1], ones(N, double).reshape(N, 1)
    cls = independent_cls()
    for cl in cls:
        print "Weights %s" % str(cl.W)
        print "Variance %s, Prior %s, %s, %s" % (str(cl.tau_bk / cl.tau_ak),
                                         str(cl.a_bk / cl.a_ak * cl.tau_bk / cl.tau_ak),
                                                 str(cl.tau_ak), str(cl.tau_bk))
        print "sum(var) %s, sum(res) %s" % (str(sum(cl.var)), str(sum(cl.res)))
    K = len(cls)
    gate = Gating(cls, N, Xf.shape[1])
    print "Training gating in %s iterations" % str(gate.update_gating(Xf))
    #print "Training gating in %s iterations" % str(gate.full_update(Xf))
    print "Variational bound: %f" % gate.var_bound(Xf)
    print exp(gate.V) / exp(gate.V).sum()
    g = plot_cls(X, y, gate)
    raw_input('Please press return to continue...\n')

    # second, custom test
    cls = [RBF1DCl(X, 1, array([0.89, 0.0316], double)),
           RBF1DCl(X, 1, array([0.18, 0.00158], double)),
           RBF1DCl(X, 1, array([0.47, 2.51e-4], double))]
    for cl in cls:
        cl.update(X, y)
    gate = Gating(cls, N, Xf.shape[1])
    print "Training gating in %d iterations" % gate.update_gating(Xf)[0]
    print "Variational bound: %f" % gate.var_bound(Xf)
    g = plot_cls(X, y, gate)
    raw_input('Please press return to continue...\n')


# run experiments from arguments
if __name__ == '__main__':
    exp_modes = {'generate': lambda: write_samples(300, 0.1),
                 'a': lambda: exp1a()}
    for argv in sys.argv[1:]:
        if not exp_modes.has_key(argv):
            print "--- Unkown experiment: %s" % argv
        else:
            print "--- Running '%s'" % argv
            exp_modes[argv]()
