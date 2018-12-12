"""A simple GA implementation for experiments with the Bayesian LCS.
"""

from numpy.random import uniform, random_integers

# argmax from http://www.daniel-lemire.com/blog/archives/2004/11/25/computing-argmax-fast-in-python/
from itertools import izip
argmax = lambda array: max(izip(array, xrange(len(array))))[1]

from cl import Gating

class Individual(object):
    """An individual in a population. Given that its chromosome is unchanged,
    its fitness is assumed to stay constant as well.
    """
    def __init__(self, chrom = None):
        """Initialises a new individual with a given chromosome. If no
        choromosome is given, then the individual is initialised randomly,
        by calling the randomize_chrom() method.
        """
        if chrom == None:
            self.randomise_chrom()
        else:
            self.chrom = chrom
        self._fitness_cache = None

    def randomise_chrom(self):
        """Randomises the chromosome of the individual.
        """
        pass

    def fitness(self):
        """Returns the fitness of the individual. If the fitness has not
        been evaluated before, then fitness_eval() is called.
        """
        if self._fitness_cache != None:
            return self._fitness_cache
        self._fitness_cache = self.fitness_eval()
        return self._fitness_cache

    def fitness_eval(self):
        """Evaluates the fitness of the individual and returns it.
        """
        pass

    def mutate(self, prob):
        """Mutates the individual, each element with the given probability.
        """
        pass

    def add(self, prob):
        """Adds an additional allele to the chromosome with given
        probability.
        """
        pass

    def crossover(self, indv):
        """Returns two new individuals as a result of the crossover of
        this individual and the given individuals. The two individuals are
        returned as (indv1, indv2).
        """
        pass

#from math import exp, cos, pi
#
#class TestIndv(Individual):
#    """Maximum at (0.5, 0.5), which is 1.0"""
#    def randomise_chrom(self):
#        self.chrom = (uniform(), uniform())
#    def fitness_eval(self):
#        x = self.chrom
#        return cos((x[0] - 0.5) * 30.0) / (1.0 + ((x[0] - 0.5) * 4.0) ** 2) * \
#               cos((x[1] - 0.5) * 30.0) / (1.0 + ((x[1] - 0.5) * 4.0) ** 2)
#    def mutate(self):
#        x = self.chrom
#        self.chrom = (max(min(x[0] + (uniform() - 0.5) * 0.1, 1.0), 0.0),
#                      max(min(x[1] + (uniform() - 0.5) * 0.1, 1.0), 0.0))
#    def crossover(self, indv):
#        x1, x2 = self.chrom, indv.chrom
#        return (TestIndv((x1[0], x2[1])), TestIndv((x2[0], x1[1])))


class GeneticAlgorithm_TS(object):
    """A genetic algorithm with tournament selection, using crossover and
    mutation operators.
    """
    def __init__(self, pop, ts, mut_prob, co_prob, add_prob):
        """Initialises the GA with the given population (a sequence of
        individuals), the given tournamen size ts, the given mutation
        probability mut_prob, the given crossover probability co_prob, and
        the addition probability. The population size of the initial
        population has to be divisible by 2 (to provide proper crossover).
        """
        self.pop, self.ts = pop, ts
        self.mut_prob, self.co_prob = mut_prob, co_prob
        self.add_prob = add_prob
        # pop_f always contains the current fitness of all individuals
        self.pop_f = [indv.fitness() for indv in pop]

    def next_gen(self):
        """Creates a new generation by selection, crossover and mutation.
        It returns the maximum, minimum and average fitness in the
        population as the triple (max, min, avg).
        """
        mut_prob, co_prob = self.mut_prob, self.co_prob
        add_prob = self.add_prob
        N = len(self.pop)
        new_pop = []
        for n in xrange(N / 2):
            # selection
            indv1, indv2 = self.select(), self.select()
            # crossover
            if uniform() <= co_prob:
                indv1, indv2 = indv1.crossover(indv2)
            # mutation
            indv1.mutate(mut_prob)
            indv2.mutate(mut_prob)
            # addition
            indv1.add(add_prob)
            indv2.add(add_prob)
            # add
            new_pop.append(indv1)
            new_pop.append(indv2)
        # evaluate new fitness vector
        new_pop_f = [indv.fitness() for indv in new_pop]
        self.pop, self.pop_f = new_pop, new_pop_f
        return (max(new_pop_f), min(new_pop_f),
                sum(new_pop_f) / float(N))

    def select(self):
        """Selects one individual by tournament selection and
        returns it.
        """
        # generate the tournament by selecting individual indicies
        t = random_integers(0, len(self.pop) - 1, self.ts)
        # get the fitnesses
        pop_f = self.pop_f
        t_f = [pop_f[i] for i in t]
        # select the highest-fitness individual and return it
        return self.pop[t[argmax(t_f)]]

    def best_indv(self):
        """Returns the currently best (highest fitness) individual in the
        population.
        """
        return self.pop[argmax(self.pop_f)]


class ClStoreIndv(Individual):
    """An individual based on a set of classifiers from a classifier store.
    """
    def __init__(self, cl_store, X, Y, Xf, chrom = None, K = 0):
        """Initialises an individual with a new chromosome. If now chromosome
        is given, then the individual is initialised with K random classifiers.
        (X, Y) is a reference to the available data. Xf is the set of
        gating features.
        """
        self.cl_store, self.X, self.Y, self.Xf = cl_store, X, Y, Xf
        if chrom == None:
            self.randomise_chrom(K)
        else:
            self.chrom = chrom
        self._fitness_cache = None

    def randomise_chrom(self, K = 1):
        """Initialises the chromosome with K classifiers.
        """
        self.chrom = [cl_store.random_cl_key() for x in xrange(K)]

    def fitness_eval(self):
        """Evaluates the fitness of the individual by fetching the
        classifiers from the classifier store, training a gating network
        with them, and returning the log of the model probability of that
        network.
        """
        X, Y, Xf = self.X, self.Y, self.Xf
        # get set of classifiers
        cls = [self.cl_store.get_cl(cl_key, X, Y)
               for cl_key in self.chrom]
        # create and train gating network
        gate = Gating(cls, X.shape[0], Xf.shape[1])
        gate.update_gating(Xf)
        self.gate = gate
        return gate.ln_model_prob(Xf)

    def mutate(self, prob):
        """Mutates the individual, each element with the given probability.
        """
        # mutation depends on the type of classifiers, needs to be overridden
        pass

    def add(self, prob):
        """Adds one additional allele to the chromosome.
        """
        if uniform() <= prob:
            self.chrom.append(self.cl_store.random_cl_key())
            
##     def crossover(self, indv):
##         """Returns two new individuals as a result of 2-point crossover of this
##         individual and the given individual. The two individuals are
##         returned as (indv1, indv2).
##         """
##         chrom1, chrom2 = self.chrom, indv.chrom
##         len1, len2 = len(chrom1), len(chrom2)
##         # get the cut points
##         cp1 = random_integers(0, len1)
##         cp2 = random_integers(0, len2)
##         # we can get individuals without classifiers in the following two
##         # constellations:
##         #  | cp1               | cp2
##         # [ A A A A A ] [ B B B ]     -> [] [ A A A A A B B B ]
##         #
##         #        cp1 |   | cp2
##         # [ A A A A A ] [ B B B ]     -> [ A A A A A B B B ] []
##         # hence we need to avoid them by either moving cp1 or cp2 by one
##         if cp1 == 0 and cp2 == len2:
##             if random_integers(0, 1) == 0: cp1 = 1
##             else: cp2 = len2 - 1
##         elif cp1 == len1 and cp2 == 0:
##             if random_integers(0, 1) == 0: cp1 = len1 - 1
##             else: cp2 = 1
##         # create the 2 new individuals
##         return (self.__class__(self.cl_store, self.X, self.Y, self.Xf,
##                             chrom1[:cp1] + chrom2[cp2:]),
##                 self.__class__(self.cl_store, self.X, self.Y, self.Xf,
##                             chrom2[:cp2] + chrom1[cp1:]))
    
    def crossover(self, indv):
        """Returns two new individuals as a result of uniform crossover of
        this individual and the given individual. The two individuals are
        returned as (indiv1, indiv2).
        """
        chromB = list(self.chrom) + list(indv.chrom)
        Ka = random_integers(1, len(chromB) - 1)
        # randomly pick Ka alleles to form chromosome of A
        chromA = []
        for k in xrange(Ka):
            i = random_integers(0, len(chromB) - 1)
            chromA.append(chromB[i])
            del(chromB[i])
        # create the 2 new individuals
        return (self.__class__(self.cl_store, self.X, self.Y, self.Xf,
                               chromA),
                self.__class__(self.cl_store, self.X, self.Y, self.Xf,
                               chromB))
        
        
