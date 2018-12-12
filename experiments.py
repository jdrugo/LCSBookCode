"""Framework module to perform experiments with the bayesian LCS.
"""

import sys
from numpy import double, array, ones, empty, hstack, inf, linspace, sqrt, \
     isnan, nan_to_num
from math import log
import Gnuplot

from cl import fact
from ga import GeneticAlgorithm_TS
from mcmc import SampleModelPosterior


def write_data(x, y, filename):
    """Writes the data to the file with given filename.
    """
    assert(x.shape[0] == y.shape[0])
    assert(len(x.shape) == 1 and len(y.shape) == 1)
    f = open(filename, 'w')
    for n in xrange(x.shape[0]):
        print >>f, "%0.6f\t%0.6f" % (x[n], y[n])
    f.close()

def write_raw_data(x, Y, filename):
    """Same as write_data, but Y is now a matrix of N rows.
    """
    assert(x.shape[0] == Y.shape[0])
    assert(len(x.shape) == 1 and len(Y.shape) == 2)
    f = open(filename, 'w')
    data_str = "\t".join(["%0.6f"] * (1 + Y.shape[1]))
    for n in xrange(x.shape[0]):
        print >>f, data_str % tuple([x[n]] + Y[n,:].tolist())
    f.close()


def read_data(filename):
    """Returns input and output matrix (X, Y) by reading the data from
    the file with the given filename.
    """
    f = open(filename, 'r')
    x, y = [], []
    for l in f.readlines():
        if l[-1] == '\n':
            l = l[:-1]
        l = l.strip()
        if l == '':
            continue
        xn, yn = map(float, l.split('\t'))
        x.append(xn)
        y.append(yn)
    X = hstack((ones(len(x), double).reshape(len(x), 1),
                array(x, double).reshape(len(x), 1)
                ))
    return (X, array(y, double).reshape(len(y), 1))


def plot_cls(X, Y, gate, filename=""):
    """Plots the data, the classifier prediction, and the mixed prediction.
    If a filename is given, then the prediction data is also written to a file
    with the given filename. The method returns the plot object. The plot is
    closed if this object is deleted. It is assumed that the second column
    of X contains the full range, and y is of shape (N, 1). The method only
    works with classifiers that model straight lines.
    """
    cls = gate.cls
    N, K = X.shape[0], len(cls)
    x = X[:,1]
    y, min_x, max_x = Y.reshape(N), x.min(), x.max()
    Xf = ones(N, double).reshape(N, 1)
    
    # get the original function
    plot_data = [ Gnuplot.Data(x.tolist(), y.tolist(), title="f(x)") ,]
    
    # get classifier predictions
    N = 100
    x = linspace(min_x, max_x, N)
    Pred = empty((N, K+3), double) # +3 for mix and its standard deviation
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

    # plot the graph
    g = Gnuplot.Gnuplot()
    g.plot(*plot_data)
    
    # write to file, if requested
    if filename != "":
        data_str = '\t'.join(["%0.6f"] * (K + 4))
        f = open(filename, 'w')
        for n in xrange(N):
            print >>f, data_str % tuple([x[n]] + list(Pred[n,:]))
    return g


def print_cls(cls):
    """Prints the classifiers in the population to the standard output.
    """
    for k in xrange(len(cls)):
        print "% 2d: %s" % (k + 1, str(cls[k]))


def GA_experiment(X, Y, Xf, epochs, Ks, cl_store, indv_class,
                  fitness_file = "", best_file = ""):
    """Performs a GA experiment by running 'epochs' epochs. The initial
    population is initialised with individuals of size K, where the Ks are
    given by the sequence Ks. If the fitness_file is given, then the best,
    worst, and average fitness, and the average number of classifiers is
    written to the file. If best_file is given, then the best final
    individual is written to this file, using plot_cls(). The tournament
    size is always 5, and mutation and crossover probability are 0.4 and
    0.8 respectively.
    """
    # create initial population
    pop = []
    for K in Ks:
        pop.append(indv_class(cl_store, X, Y, Xf,
                              [cl_store.random_cl_key() for k in xrange(K)]))
    # initialise GA
    GA = GeneticAlgorithm_TS(pop, 5, 0.4, 0.4, 0.00)
    gr = None
    fitnesses = empty((epochs, 4), double)
    best_varbound, best_cls = -inf, None
    
    # run epochs
    print "Running epoch %6d" % 0,
    sys.stdout.flush()
    for epoch in xrange(epochs):
        #print "\033[7D%6d" % (epoch + 1),
        #sys.stdout.flush()
        
        # create new populations and get stats
        fitnesses[epoch,0:3] = GA.next_gen()
        Ks = array([len(indv.chrom) for indv in GA.pop], double)
        fitnesses[epoch,3] = Ks.mean()

        # print population structure
        pop, pop_f = GA.pop, GA.pop_f
        for k in xrange(len(pop)):
            print pop_f[k], pop[k].chrom
        print "----- %d" % (epoch + 1)

        # store best individual
        if best_varbound < fitnesses[epoch, 0]:
            best_cls = GA.best_indv()
            best_varbound = fitnesses[epoch, 0]

        # generate graph every 10th epoch
        if epoch % 10 == 0:
            del(gr)
            gr = plot_cls(X, Y, GA.best_indv().gate)
    print
        
    # write fitnesses to file
    if fitness_file:
        fitnesses[isnan(fitnesses)] = -inf
        fitnesses = nan_to_num(fitnesses)
        f = open(fitness_file, 'w')
        print >>f,"# Epoch, Max, Min, Avg fitness, average K"
        for epoch in xrange(epochs):
            print >>f, "%d\t%0.6f\t%0.6f\t%0.6f\t%0.6f" % \
                  (epoch + 1, fitnesses[epoch, 0], fitnesses[epoch, 1],
                   fitnesses[epoch, 2], fitnesses[epoch, 3])
    # write best individual to file
    if best_file:
        gr = plot_cls(X, Y, best_cls.gate, best_file)
    else:
        gr = plot_cls(X, Y, best_cls.gate)
        
    # print best individual
    print "Best individual:"
    print_cls(best_cls.gate.cls)
    print "Variational bound: %0.6f" % best_varbound

    raw_input('Please press return to continue...\n')
    del gr


def MCMC_experiment(X, Y, Xf, inner_steps, outer_steps, del_add_prob,
                    K, cl_store, varbound_file = "", best_file = ""):
    """Performs an MCMC experiment by running outer_steps runs of
    inner_steps steps each, and reinitialising the population before each
    run with K classifiers. The probabiliy for adding and deleting classifiers
    is given by del_add_prob, and the classifier store cl_store is used.
    If varbound_file is given, then the variational bound of each step, as
    well as the current number of classifiers is written to that file.
    If best_file is given, then the best set of classifiers is written
    to that file.
    """
    best_cls, best_varbound = None, -inf
    varbound_plot = Gnuplot.Gnuplot()
    cls_plot = None
    varbounds = empty((inner_steps * outer_steps, 2), double)
    total_actions, total_rejects = [0, 0, 0], [0, 0, 0]
    step = 0
    for outer in xrange(outer_steps):
        # use sys to get immediate output
        print "Running outer loop %d, inner %6d" % (outer + 1, 0),
        sys.stdout.flush()
        rejects, accepts = [0, 0, 0], [0, 0, 0]
        best_inner_cls, best_inner_varbound = None, -inf
        # initialise sampler
        cls = [cl_store.random_cl(X, Y) for k in xrange(K)]
        sampler = SampleModelPosterior(del_add_prob, cls, cl_store, X, Y, Xf)
        for inner in xrange(inner_steps):
            if inner % 10 == 0:
                print "\033[7D%6d" % inner,
                sys.stdout.flush()
            # perform next step
            act, accepted = sampler.next()
            # create some stats
            total_actions[act] += 1
            if accepted == False:
                rejects[act] += 1
                total_rejects[act] += 1
            else:
                accepts[act] += 1
                # store cls if better (inner loop)
                if sampler.var_bound > best_inner_varbound:
                    best_inner_cls = sampler.gate
                    best_inner_varbound = sampler.var_bound
            varbounds[step, 0] = sampler.var_bound
            varbounds[step, 1] = len(sampler.cls)
            step += 1
        # store cls if better (outer loop)
        if best_inner_varbound > best_varbound:
            best_cls = best_inner_cls
            best_varbound = best_inner_varbound
        # print stats
        print
        print "       Accepted Rejected"
        print "Change %8d %8d" % (accepts[0], rejects[0])
        print "Remove %8d %8d" % (accepts[1], rejects[1])
        print "Add    %8d %8d" % (accepts[2], rejects[2])
        print_cls(best_inner_cls.cls)
        print "Variational bound: %0.6f" % best_inner_varbound
        print
        # plot graphs
        del(cls_plot)
        cls_plot = plot_cls(X, Y, best_inner_cls)
        varbound_plot.plot(
            Gnuplot.Data(varbounds[:step,0], with='lines'),
            Gnuplot.Data(varbounds[:step,1] * 10, with='lines'))

    # need to remove previous plot
    del(cls_plot)

    # write varbounds to file
    if varbound_file:
        f = open(varbound_file, 'w')
        print >>f,"# Step, Varbound, K"
        for step in xrange(inner_steps * outer_steps):
            print >>f, "%d\t%0.6f\t%0.6f" % \
                  (step + 1, varbounds[step, 0], varbounds[step, 1])
            
    # write best population to file
    if best_file:
        cls_plot = plot_cls(X, Y, best_cls, best_file)
    else:
        cls_plot = plot_cls(X, Y, best_cls)

    # write stats to standard output
    print "       Total    Rejected"
    print "Change %8d %4.1f%%" % (total_actions[0], float(total_rejects[0]) /
                                  total_actions[0] * 100.0)
    print "Remove %8d %4.1f%%" % (total_actions[1], float(total_rejects[1]) /
                                  total_actions[1] * 100.0)
    print "Add    %8d %4.1f%%" % (total_actions[2], float(total_rejects[2]) /
                                  total_actions[2] * 100.0)
    print "Total  %8d %4.1f%%" % (total_actions[0] + total_actions[1] +
                                  total_actions[2],
                                  float(total_rejects[0] + total_rejects[1] +
                                        total_rejects[2]) /
                                  (total_actions[0] + total_actions[1] +
                                   total_actions[2]) * 100.0)
    print "Overall best population:"
    print_cls(best_cls.cls)
    print "Variational bound: %0.6f" % best_varbound
    print "L(q) - ln K!: %0.6f" % (best_varbound - log(fact(len(best_cls.cls))))
    print
    raw_input('Please press return to continue...\n')

    
