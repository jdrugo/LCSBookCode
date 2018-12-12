"""Collection of classifiers with different matching functions.
"""

from numpy import double, array, exp
from numpy.random import random_integers, normal, uniform

from cl import Classifier
from ga import ClStoreIndv


class ClassifierStore(dict):
    """A collection of pre-trained classifiers. This class required some
    discritisation of the classifier space, to avoid storing an infinite
    number of classifiers.
    """
    def new_cl(self, X, Dy, cl_key):
        """Returns a new classifier whose matching parameters conform
        to the given classifier key.
        """
        pass

    def random_cl_key(self):
        """Returns a random classifier key.
        """
        pass

    def get_cl(self, cl_key, X, Y):
        """Returns a classifier with the given key. If the classifier is not
        yet in the store, then a new one is created based on the given key and
        trained with the given data, in which case newcl() is called.
        """
        try:
            return self.__getitem__(cl_key)
        except:
            pass
        cl = self.new_cl(X, Y.shape[1], cl_key)
        if cl.update(X, Y) < 1:
            print "Classifier '%s' training exceeded maximum iteration limit" \
                  % str(cl_key)
        self.__setitem__(cl_key, cl)
        return cl

    def random_cl(self, X, Y):
        """Returns a randomly generated, and trained classifier.
        """
        return self.get_cl(self.random_cl_key(), X, Y)


class RBF1DCl(Classifier):
    """A 1D Radial basis function matching classifier. The two matching
    parameters are the centre of the classifier and its spread. The spread
    is equivalent to the variance of the Gaussian.
    """
    def matches(self, x):
        """Returns the matching for input x, based on a Gaussian-like
        radial basis funcion. Only the second element of the two-element
        input is considered, as the first is usually the bias term.
        """
        # m = exp( 1/(2var) * (x - mu)^2 )
        return exp(-0.5 / self.m_param[1] * (x[1] - self.m_param[0]) ** 2.0)

    def __str__(self):
        """Returns the matching function as a string
        'Mean 0.000, Variance 0.000'
        """
        return "Mean %6.4f, %6.4f" % (self.m_param[0], self.m_param[1])


class RBF1DClStore(ClassifierStore):
    """A classifier store for RBF1DCl classifiers. The classifier key is the
    pair (a, b), where a determines the mean, and b the spread of the
    classifier. a is in [0, 100], where 0 determines the lower bound, and
    100 determines the upper bound of the mean. b determines the variance
    by 1/(10^(b/10)) and has to be in range [0, 50].
    """
    def __init__(self, lm = 0.0, um = 1.0):
        """Initialises the store with the given lower and upper bound for
        the mean.
        """
        self._lm, self._sm = lm, (um - lm) / 100.0

    def new_cl(self, X, Dy, cl_key):
        """Returns a new classifier for the given key.
        """
        return RBF1DCl(X, Dy,
                       array([self._lm + self._sm * cl_key[0],
                              1.0 / 10.0 ** (cl_key[1] / 10.0)], double))

    def random_cl_key(self):
        """Returns a random classifier key.
        """
        return (random_integers(0, 100), random_integers(0, 50))


class RBF1DIndv(ClStoreIndv):
    """An individual for use in a GA, describing a set of RBF1D classifiers.
    """
    def mutate(self, prob):
        """Mutates each element of the individual with the given probability.
        Each mean is mutated by N(0, 10), and each spread by N(0, 5).
        """
        chrom = self.chrom
        for k in xrange(len(chrom)):
            if uniform() <= prob:
                cl_key = chrom[k]
                chrom[k] = (max(0, min(100,
                                       int(cl_key[0] + normal(0.0, 10.0)))),
                            max(0, min(50,
                                       int(cl_key[1] + normal(0.0, 5.0)))))


class SoftIntervalCl(Classifier):
    """A 1D soft interval matching classifier. The two matching
    parameters are the two bound of the interval, where the boundaries are
    softened by a Gaussian, of which one standard deviation is within the
    interval boundaries.
    """
    def matches(self, x):
        """Returns the matching for input x, based on a Gaussian-like
        radial basis funcion. Only the second element of the two-element
        input is considered, as the first is usually the bias term.
        """
        m_param = self.m_param
        b = m_param[1] - m_param[0]
        l, u = m_param[0] + 0.05666 * b, m_param[1] - 0.05666 * b
        if x[1] < l:
            return exp(- 0.5 / (0.0662 * b) ** 2 * (x[1] - l) ** 2)
        elif x[1] > u:
            return exp(- 0.5 / (0.0662 * b) ** 2 * (x[1] - u) ** 2)
        else:
            return 1.0

    def __str__(self):
        """Returns the matching function as a string
        '[0.000 - 0.000]'
        """
        return "[%6.4f - %6.4f]" % (self.m_param[0], self.m_param[1])

        
class SoftIntervalClStore(ClassifierStore):
    """A classifier store for soft interval classifiers. The classifier key
    is the pair (a, b), where a determines the lower boundary, and b the
    upper boundary. Both values are in [0, 100], where 0 is the lowest,
    and 100 the highest value of the range of the input.
    """
    def __init__(self, lm = 0.0, um = 1.0):
        """Initialises the store with the given lower and upper bound for
        the input.
        """
        self._lm, self._sm = lm, (um - lm) / 100.0

    def new_cl(self, X, Dy, cl_key):
        """Returns a new classifier for the given key.
        """
        return SoftIntervalCl(X, Dy,
                              array([self._lm + self._sm * cl_key[0],
                                     self._lm + self._sm * cl_key[1]], double))

    def random_cl_key(self):
        """Returns a random classifier key.
        """
        a, b = random_integers(0, 100), random_integers(0, 100)
        # make sure that b > a
        if a > b:
            return (b, a)
        return (a, b)

        
class SoftInterval1DIndv(ClStoreIndv):
    """An individual for use in a GA, describing a set of soft interval
    classifiers.
    """
    def mutate(self, prob):
        """Mutates each element of the individual with the given probability.
        The boundaries of each element are mutated by N(0, 10).
        """
        chrom = self.chrom
        for k in xrange(len(chrom)):
            if uniform() <= prob:
                cl_key = chrom[k]
                cl_key = (max(0, min(100,
                                     int(cl_key[0] + normal(0.0, 10.0)))),
                          max(0, min(100,
                                     int(cl_key[1] + normal(0.0, 10.0)))))
                # spread interval if 0 width
                if cl_key[0] == cl_key[1]:
                    cl_key = (max(0, min(100, cl_key[0] - 1)),
                              max(0, min(100, cl_key[1] + 1)))
                # make sure that it is correctly ordered
                if cl_key[0] > cl_key[1]:
                    chrom[k] = (cl_key[1], cl_key[0])
                else:
                    chrom[k] = cl_key
