from numpy import array, arange, ones, power, hstack, log, pi
from numpy.random import randn
import cl

# output vector
N = 100
noise = randn(N) * 0.3 ** 0.5
x = arange(0, N) / float(N)
Y1 = (x * 2.0 + 3.0).reshape(N, 1)
Y2 = ((x * 2.0 + noise) + 3.0).reshape(N, 1)
Y3 = (x * 2.0 + 0.5 * power(x, 2.0) + 3.0).reshape(N, 1)
Y4 = (x * 2.0 + 0.5 * power(x, 2.0) + 3.0 + noise).reshape(N, 1)
# multi-dimensional output
Y5 = hstack((Y2, Y4))

# build test matrix
x.shape = (N, 1)
X1 = hstack((ones((N, 1)), x))
X2 = hstack((ones((N, 1)), x, power(x, 2.0)))

def run_test(X, Y, test_name):
    print test_name
    c = cl.Classifier(X, Y.shape[1], None)
    print "Trained in %d iterations" % c.update(X, Y)[0]
    print "Weights %s\nVariance %s, Prior %s" % \
          (str(c.W), str(c.tau_bk / c.tau_ak), str(c.a_bk / c.a_ak))
    c.var_bound_test(ones(N))
    print "Variational bound %s" % str(c.var_bound(ones(N)))
    print

run_test(X1, Y1, "Linear approximation 3 + 2x")
run_test(X1, Y2, "Linear approximation 3 + 2x + N(0, 0.3)")
run_test(X2, Y1, "Linear approximation 3 + 2x, extra feature")
run_test(X2, Y2, "Linear approximation 3 + 2x + N(0, 0.3), extra feature")
run_test(X2, Y3, "Square approximation 3 + 2x + 0.5x^2")
run_test(X2, Y4, "Square approximation 3 + 2x + 0.5x^2 + N(0, 0.3)")
run_test(X1, Y4, "Square approximation 3 + 2x + 0.5x^2 + N(0, 0.3), missing feature")
run_test(X1, Y5, "Linear approximation 3 + 2x + N(0, 0.3) and\n                     3 + 2x + 0.5x^2 + N(0, 0.3)")
