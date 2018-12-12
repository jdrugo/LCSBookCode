"""Monte carlo markov chain method to sample the model posterior.
"""

from numpy import log, inf
from numpy.random import randint, uniform

from cl import Gating


class SampleModelPosterior(object):
    """Class to sample the model posterior, using the Metropolis-Hasting
    algorithm.
    """
    # action constants
    A_CHANGE = 0
    A_REMOVE = 1
    A_ADD    = 2
    
    def __init__(self, add_remove_p, cls, cl_store, X, Y, Xf):
        """Initialises the sampler with the given probability to remove
        and add classifier. If the probability is 0.25, for example, the
        classifiers are added with probability 0.25, removed with
        probability 0.25, and changed with probability 0.5. The initial
        set of classifiers is given by cls. The classifier store should be
        an object of a subclass of ClassifierStore. The initial classifier
        set is trained on the given data (X, y).
        """
        self.c_p = 1.0 - 2.0 * add_remove_p
        self.r_p = 1.0 - add_remove_p
        self.store = cl_store
        self.cls = cls
        self.X, self.Y, self.Xf = X, Y, Xf
        K, N, Dv = len(self.cls), X.shape[0], Xf.shape[1]
        # create and train gate and get variational bound
        self.gate = Gating(cls, N, Dv)
        self.gate.update_gating(Xf)
        self.var_bound = self.gate.var_bound(Xf)

    def next(self):
        """Perform single step in Markov chain, given the data. This method
        first chooses amongst (change, remove, add) according to the preset
        probabilities and then creates a candiate solution for either action.
        If also calculates the acceptance probability and keeps the solution
        if it was accepted. The method returns the tuple (action_id, accepted),
        where action_id is one of A_CHANGE, A_REMOVE or A_ADD, and accpeted
        indicates if the change was accepted.
        """
        store = self.store
        # candidate classifier set
        cand_cls = self.cls[:]
        X, Y, Xf = self.X, self.Y, self.Xf
        K, N, Dv = len(cand_cls), X.shape[0], Xf.shape[1]
        # choose action to perform
        act = uniform()
        if act <= self.c_p:
            act_id = SampleModelPosterior.A_CHANGE
            # change randomly chosen classifiers
            cand_cls[randint(0, K)] = store.random_cl(X, Y)
            # train new gating network
            cand_gate = Gating(cand_cls, N, Dv)
            cand_gate.update_gating(Xf)
            cand_var_bound = cand_gate.var_bound(Xf)
            # get acceptance probability
            ln_accept_prob = min(cand_var_bound - self.var_bound, 0)
        elif act <= self.r_p:
            act_id = SampleModelPosterior.A_REMOVE
            # only remove if we have classifiers
            if len(cand_cls) == 1:
                # removing the only classifier will certainly be rejected
                ln_accept_prob = -inf
            else:
                # remove randomly chosen classifier
                del(cand_cls[randint(0, K)])
                # train new gating network
                cand_gate = Gating(cand_cls, N, Dv)
                cand_gate.update_gating(Xf)
                cand_var_bound = cand_gate.var_bound(Xf)
                # get acceptance probability
                ln_accept_prob = min(2 * log(K) + cand_var_bound -
                                     self.var_bound,
                                     0)
        else:
            act_id = SampleModelPosterior.A_ADD
            # add classifier
            cand_cls.append(store.random_cl(X, Y))
            # train new gating network
            cand_gate = Gating(cand_cls, N, Dv)
            cand_gate.update_gating(Xf)
            cand_var_bound = cand_gate.var_bound(Xf)
            # get acceptance probability
            ln_accept_prob = min(cand_var_bound - self.var_bound -
                                 2 * log(K + 1), 0)
        # accept?
        if log(uniform()) < ln_accept_prob:
            # yes -> store new classifier set
            self.cls, self.gate = cand_cls, cand_gate
            self.var_bound = cand_var_bound
            return (act_id, True)
        else:
            return (act_id, False)
