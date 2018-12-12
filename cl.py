"""Bayesian LCS, using variational bayesian approach.
"""

from numpy import double, array, zeros, ones, identity, empty, vstack, \
     dot, sum, log, power, concatenate, abs, max, exp, clip, nan_to_num, \
     pi, inf, isnan
     
from numpy.linalg import pinv, det, LinAlgError
from numpy.random import uniform, randint
from scipy.special import gammaln, psi
from math import log as math_log

# fact from http://importantshock.wordpress.com/2006/11/03/one-line-factorial-function-in-python/
import operator
def fact(x): return reduce(operator.mul, xrange(2, x+1), 1)

# constants for later use
ln2pi = log(2.0 * pi)
double_max = nan_to_num(double(inf))
double_min = nan_to_num(double(-inf))
ln_double_max = log(double_max)
exp_min = double(-708.3964) # for double=float64 type


class Classifier(object):
    """A single classifier, if used needs to have matches() overridden.
    """
    # noise prior Gam(tau_a, tau_b) parameters
    tau_a = 1.e-2
    tau_b = 1.e-4
    # weight vector hyperprior Gam(a_a, a_b) parameters
    a_a = 1.e-2
    a_b = 1.e-4
    # constant for variational bound
    L_tau_const = -gammaln(tau_a) + tau_a * log(tau_b)
    L_a_const = -gammaln(a_a) + a_a * log(a_b)
    varbound_const = - gammaln(tau_a) + tau_a * (log(tau_b) + ln2pi) \
                     - gammaln(a_a) + a_a * log(a_b)
    # stopping criteria for classifier update
    max_iter = 20
    max_dL = 1.e-4

    def __init__(self, X, Dy, m_param):
        """Initialises classifier with given input matrix X, for Dy outputs.
        The classifier is not trained, but its matching vector is initialised,
        using the parameter vector m_param.
        """
        N, Dx = X.shape
        # weight matrix model N(wj, Sig), wj is rows of W
        self.W = zeros((Dy, Dx), double)
        self.Sig = zeros((Dx, Dx), double)
        # noise precision model Gam(tau_ak, tau_bk)
        self.tau_ak = Classifier.tau_a
        self.tau_bk = Classifier.tau_b
        # weight prior model Gam(a_ak, a_bk)
        self.a_ak = Classifier.a_a
        self.a_bk = Classifier.a_b
        # evaluate matching
        self.m_param = m_param
        self.m = self.init_matching(X, m_param)
        # cached values
        self.res = zeros(N, double)
        self.var = zeros(N, double)
        self.ln_Sig_det = -inf

    def matches(self, x):
        """Returns the degree to which the classifier matches the input.
        In the base class it always returns 1.0.
        """
        return 1.0

    def __str__(self):
        """Returns a string representation of its matching function.
        In the base class it always returns 'all'.
        """
        return "all"

    def init_matching(self, X, m_param):
        """Initialises the matching vector for the given input matrix by
        classing matches() for each input in the input matrix X. m_param is
        the matching parameter vector.
        """
        return array(
            [self.matches(X[n,:]) for n in xrange(X.shape[0])],
            double)

    def update(self, X, Y, resp = None):
        """Updates the classifier model with given input matrix X, output
        matrix Y, and responsibilities resp. If the responsibilities are not
        given, then the classifier's matching vector is used instead. The
        method returns (interations, L) where iterations is the number of
        interations that the update required, or a negative number if the
        update didn't converge within the maximum number of updates. L is the
        variational bound of the classifier at the last iteration, and
        convergence is tested by monitoring the change of this value.
        """
        if resp == None:
            resp = self.m
        # dimensions
        N, Dx = X.shape
        Dy = Y.shape[1]
        DxDy = Dx * Dy
        # local cache
        tau_a, tau_b = Classifier.tau_a, Classifier.tau_b
        a_a, a_b = Classifier.a_a, Classifier.a_b
        L_tau_const, L_a_const = Classifier.L_tau_const, Classifier.L_a_const
        max_dL, max_iter = Classifier.max_dL, Classifier.max_iter
        # responsibility-weighted input and output
        resp_mat = resp.reshape((resp.shape[0], 1))
        Xk, Yk = X * resp_mat, Y * resp_mat
        resp_sum = sum(resp)

        # initial values
        a_ak, a_bk = self.a_ak, self.a_bk
        tau_ak, tau_bk, W = self.tau_ak, self.tau_bk, self.W
        L, dL, i = -inf, max_dL + 1.0, 0
        while (i < max_iter) and (dL > max_dL):
            # update weight vector model
            prec_mat = identity(Dx, double) * (a_ak / a_bk) + dot(X.T, Xk)
            Sig = pinv(prec_mat)
            W = dot(dot(Y.T, Xk), Sig)
            ln_Sig_det = log(det(Sig))
            # update noise precision model
            tau_ak = tau_a + 0.5 * resp_sum
            tau_bk = tau_b + 0.5 / float(Dy) * (sum(Yk * Y) -
                                                sum(W * dot(W, prec_mat)))
            # alternative (more expensive) to noise precision model update
            #res = sum(power(Y - dot(X, W.T), 2.0), 1)
            #tau_bk = tau_b + 0.5 / float(Dy) * (dot(resp, res) +
            #                                    a_ak / a_bk * sum(W * W))
            # update weight prior model
            a_ak = a_a + 0.5 * DxDy
            a_bk = 0.5 * (tau_ak / tau_bk * sum(W * W) + Dy * Sig.trace())
            # residual and variances (for the variational bound)
            res = sum(power(Y - dot(X, W.T), 2.0), 1)
            var = sum(X * dot(X, Sig), 1)
            # variational bound components
            prev_L = L
            psi_tau_ak, ln_tau_bk = psi(tau_ak), log(tau_bk)
            E_tau = tau_ak / tau_bk
            L_tau = float(Dy) * (L_tau_const + (tau_a - tau_ak) * psi_tau_ak -
                                 tau_a * ln_tau_bk - tau_b * E_tau +
                                 gammaln(tau_ak) + tau_ak)
            L_Y = 0.5 * float(Dy) * \
                  (psi_tau_ak - ln_tau_bk - ln2pi) * resp_sum - \
                  0.5 * dot(resp, (E_tau * res + float(Dy) * var))
            L_Wa = L_a_const + gammaln(a_ak) - a_ak * log(a_bk) + \
                   0.5 * (DxDy + float(Dy) * ln_Sig_det)
            L = L_tau + L_Y + L_Wa
            # monitor change of variational bound
            dL = L - prev_L
            if dL < 0.0:
                raise Exception, "Variational bound decreased by %f" % -dL
            # next iteratoin
            i += 1

        # copy values into object variables
        self.Sig, self.W = Sig, W
        self.tau_ak, self.tau_bk = tau_ak, tau_bk
        self.a_ak, self.a_bk = a_ak, a_bk
        # cache residuals and uncertainties, and covariance determinant
        self.res = sum(power(Y - dot(X, W.T), 2.0), 1)
        self.var = sum(X * dot(X, Sig), 1)
        self.ln_Sig_det = ln_Sig_det
        # return number of iterations taken
        if (dL > max_dL):
            return (-1, L)
        else:
            return (i, L)

    def pred(self, x):
        """Returns the prediction mean vector for the given input x.
        """
        return dot(self.W, x)

    def pred_var(self, x):
        """Returns the prediction mean vector and variance for the given
        input x, as the tuple (mean vector, variance). The variance is
        the same for all elements of the output vector.
        """
        var = 2 * self.tau_bk / (self.tau_ak - 1.0) * \
              (1.0 + dot(dot(x, self.Sig), x))
        return (dot(self.W, x), var)

    def var_bound(self, resp = None):
        """Returns the variational bound. If not responsibilities resp are
        given, then the matching values are taken instead.
        """
        if resp == None:
            resp = self.m
        # cache some values locally
        Dy, Dx = self.W.shape
        tau_a, tau_b = Classifier.tau_a, Classifier.tau_b
        tau_ak, tau_bk = self.tau_ak, self.tau_bk
        a_a, a_b = Classifier.a_a, Classifier.a_b
        a_ak, a_bk = self.a_ak, self.a_bk
        psi_tau_ak, ln_tau_bk = psi(tau_ak), log(tau_bk)
        E_tau = tau_ak / tau_bk
        # E(ln p(Y|Wk, Tk, zk))
        E_Y = 0.5 * float(Dy) * (psi_tau_ak - ln_tau_bk - ln2pi) * sum(resp) -\
              0.5 * dot(resp, E_tau * self.res + float(Dy) * self.var)
        # related to alpha and W
        E_Wa = Classifier.L_a_const + \
               gammaln(a_ak) - a_ak * log(a_bk) + \
               0.5 * Dy * (Dx + self.ln_Sig_det)
        # related to tau
        E_tau = Dy * (Classifier.L_tau_const +
                      (tau_a - tau_ak) * psi_tau_ak -
                      tau_a * ln_tau_bk - tau_b * E_tau +
                      gammaln(tau_ak) + tau_ak)
        return E_Y + E_Wa + E_tau

    def var_bound_test(self, resp = None):
        """Returns the variational bound and prints all of its components to
        the standard output.
        """
        if resp == None:
            resp = self.m
        # cache some values locally
        Dy, Dx = self.W.shape
        DxDy = float(Dx * Dy)
        tau_a, tau_b = Classifier.tau_a, Classifier.tau_b
        tau_ak, tau_bk = self.tau_ak, self.tau_bk
        W, Sig, ln_Sig_det = self.W, self.Sig, self.ln_Sig_det
        a_a, a_b = Classifier.a_a, Classifier.a_b
        a_ak, a_bk = self.a_ak, self.a_bk
        psi_tau_a, ln_tau_b = psi(tau_a), log(tau_b)
        psi_tau_ak, ln_tau_bk = psi(tau_ak), log(tau_bk)
        psi_a_a, ln_a_b = psi(a_a), log(a_b)
        psi_a_ak, ln_a_bk = psi(a_ak), log(a_bk)
        E_tau, E_a = tau_ak / tau_bk, a_ak / a_bk
        # E(ln p(Y|Wk, Tk, zk))
        E_p_Y = 0.5 * float(Dy) * \
                (psi_tau_ak - ln_tau_bk - ln2pi) * sum(resp) - \
                0.5 * dot(resp, E_tau * self.res + float(Dy) * self.var)
        # E(ln p(Wk, Tk | Ak))
        E_p_W_T = 0.5 * DxDy * (psi_a_ak - ln_a_bk +
                                psi_tau_ak - ln_tau_bk - ln2pi) - \
                  0.5 * E_a * (E_tau * sum(W * W) + float(Dy) * Sig.trace()) +\
                  float(Dy) * (- gammaln(tau_a) + tau_a * ln_tau_b +
                               (tau_a - 1.0) * (psi_tau_ak - ln_tau_bk) -
                               tau_b * E_tau)
        # E(ln p(Ak))
        E_p_A = - gammaln(a_a) + a_a * ln_a_b + \
                (a_a - 1.0) * (psi_a_ak - ln_a_bk) - a_b * E_a
        # E(ln q(Wk, Tk))
        E_q_W_T = 0.5 * DxDy * (psi_tau_ak - ln_tau_bk - ln2pi - 1.0) - \
                  0.5 * float(Dy) * ln_Sig_det + \
                  float(Dy) * (- gammaln(tau_ak) +
                               (tau_ak - 1.0) * psi_tau_ak +
                               ln_tau_bk - tau_ak)
        # E(ln q(Ak))
        E_q_A = - gammaln(a_ak) + (a_ak - 1.0) * psi_a_ak + ln_a_bk - a_ak
        L = E_p_Y + E_p_W_T + E_p_A - E_q_W_T - E_q_A
        # output and return
        print "E(ln p(Y | Wk, Tk, zk)) = %8.3f" % E_p_Y
        print "   E(ln p(Wk, Tk | Ak)) = %8.3f" % E_p_W_T
        print "            E(ln p(Ak)) = %8.3f" % E_p_A
        print "      - E(ln q(Wk, Tk)) = %8.3f" % -E_q_W_T
        print "          - E(ln q(Ak)) = %8.3f" % -E_q_A
        print "                  Lk(q) = %8.3f" % L
        return L
        

class Gating(object):
    """The gating network, used to combine the prediction of the different
    classifiers. It can be trained either independently of the classifiers,
    or in combination with them.
    """
    # prior model for gating weight prior
    b_a = 1.e-2
    b_b = 1.e-4
    # constant for variational bound
    L_b_const = -gammaln(b_a) + b_a * log(b_b)
    # convergence criteria for IRLS and full update
    irls_max_iter = 40
    irls_max_dKL = 1.e-8
    max_iter = 40
    max_dL = 1.e-2
    
    def __init__(self, cls, N, Dv):
        """Initialises the gating network for the given set cls of classifiers.
        N is the number of training samples available, and Dv is the size
        of the gating feature vector.
        """
        self.cls = cls
        K = len(cls)
        # responsibilities are initialised equiprobably
        self.R = ones((N, K), double) / double(K)
        # gating weight vectors are columns of V
        self.V = ones((Dv, K), double)
        # prior model in weight vector (beta). Each row represents the
        # parameters (a, b) for the kth gating weight prior
        self.b = ones((K, 2), double)
        self.b[:,0] *= Gating.b_a
        self.b[:,1] *= Gating.b_b
        # cached values for computing variational bound
        self.ln_cov_det = 0.0
        self.cov_Tr = zeros(K, double)

    def gating_matrix(self, Xf, V = None):
        """Returns the matrix G of gating weights for the given feature
        matrix Xf, with each column corresponding to the gating weights
        for each state of one classifier. If V is given, then its values
        are used rather than the gating weights of the gating network.
        """
        if V == None:
            V = self.V
        cls = self.cls
        K = len(cls)
        # limit the activation that we won't get an over/underflow when
        # computing the exponential. From below we have to make sure
        # that it is larger than log(0.0) such that exp(it) > 0. From above, it
        # needs to be smaller than log(double_max / K) as we need to make
        # sure that sum(exp(it)) <= K * exp(it) < inf (for the normalisation
        # step)
        G = dot(Xf, V)
        G = exp(clip(G, exp_min, ln_double_max - log(K)))
        for k in xrange(K):
            # apply matching to k'th column
            G[:,k] *= cls[k].m
        # normalise G
        G /= sum(G, 1).reshape(G.shape[0], 1)
        # due to matching it could be that the gating for particular states
        # was 0.0 for all classifiers, causing 0.0 / 0.0 = nan. Hence, we are
        # going to remove those by gating these states equally to all
        # classifiers.
        G[isnan(G)] = 1.0 / float(K)
        return G

    def hessian(self, Xf, G):
        """Return the hessian matrix for the given gating feature matrix Xf,
        and the gating weight matrix G.
        """
        Dv, K = self.V.shape
        N = Xf.shape[0]
        b_identity = identity(Dv, double)
        E_b = self.b[:,0] / self.b[:,1]
        H = empty((K * Dv,) * 2, double)
        # fill hessian block-wise
        for k in xrange(K):
            gk = G[:,k]
            kb = k * Dv
            # create block elements (k,j) and (j,k) (they are the same)
            for j in xrange(k):
                # get - (gk * gj) (element-wise)
                gkj = - gk * G[:,j]
                Hkj = dot(Xf.T, Xf * gkj.reshape(N, 1))
                # distribute block
                jb = j * Dv
                H[kb:kb+Dv, jb:jb+Dv] = Hkj
                H[jb:jb+Dv, kb:kb+Dv] = Hkj
            # create diagonal entry (k,k)
            gkk = gk * (1.0 - gk)
            Hkk = dot(Xf.T, Xf * gkk.reshape(N, 1)) + b_identity * E_b[k]
            H[kb:kb+Dv, kb:kb+Dv] = Hkk
        return H

    def update_weights(self, Xf):
        """Applies the IRLS algorithm to update the gating weights. The
        Newton-Raphson steps are preformed until the KL-distance between
        the desired responsibilities and the real gating converges, or until
        the maximum number of iterations is reached. The method returns
        (iterations, KL), where interations is the number of iterations
        required, and KL is the Kullback-Leibler divergence after the
        last update. If the maximum number of iterations is exceeded,
        then -1 is returned for the iterations.
        """
        R = self.R
        N, Dv = Xf.shape
        b_mean = self.b[:,0] / self.b[:,1]
        V = self.V
        K = V.shape[1]

        # iterate until onvergence
        max_iter, max_dKL = Gating.irls_max_iter, Gating.irls_max_dKL
        i, KL, dKL = 0, -inf, max_dKL + 1.0
        G = self.gating_matrix(Xf, V)
        while (dKL > max_dKL) and (i < max_iter):
            # target vector if column-ravelled (Xf' (G - R) - E[b] * V),
            # where last product is element-wise for each row in V
            e = (dot(Xf.T, G - R) + V * b_mean).T.ravel()
            # get inverted hessian
            try:
                H_inv = pinv(self.hessian(Xf, G))
            except LinAlgError, e:
                print "LinAlgError on computing pseudo-inverse of hessian"
                raise e
            # update gating weight vector
            V -= dot(H_inv, e).reshape(K, Dv).T
            # get gating vector for updated V (for KL divergence)
            G = self.gating_matrix(Xf, V)
            # update Kullback-Leibler divergence between G and R
            prev_KL = KL
            KL = sum(R * nan_to_num(log(G / R)))
            # as the IRLS is not a variational algorithm, it is not
            # guaranteed to converge monotonically
            dKL = abs(KL - prev_KL)
            # next iteration
            i += 1

        # update gating weight model
        self.V = V
        # get new hessian for updated gating weights to
        # compute values that are required for the variational bound.
        # We cannot use the last hessian from the IRLS iteration, as the
        # weights have been updated since then.
        try:
            H_inv = pinv(self.hessian(Xf, G))
        except LinAlgError, e:
            print "LinAlgError on computing pseudo-inverse of hessian"
            raise e
        self.ln_cov_det = log(det(H_inv))
        cov_Tr = self.cov_Tr
        for k in xrange(K):
            kb = k * Dv
            cov_Tr[k] = H_inv[kb:kb+Dv, kb:kb+Dv].trace()
        # return number of iterations
        if dKL > max_dKL:
            return (-1, KL)
        else:
            return (i, KL)

    def update_resp(self, Xf):
        """Updates the responsibilities matrix, based on the current
        goodness-of-fit of the classifiers, and the current gating weight
        vectors. Xf is the gateing feature matrix.
        """
        R, cls = self.R, self.cls
        Dy = float(cls[0].W.shape[0])
        
        # fill R with goodness-of-fit data from classifiers
        for k in xrange(R.shape[1]):
            cl = cls[k]
            tau_ak, tau_bk = cl.tau_ak, cl.tau_bk
            # k'th column is exp( Dy/2 E[ln Tk] - 1/2 (E[Tk] res + Dy var) )
            R[:,k] = exp(0.5 * (Dy * (psi(tau_ak) - log(tau_bk)) -
                                (tau_ak / tau_bk) * cl.res + Dy * cl.var))
        # multiply with current gating
        R *= self.gating_matrix(Xf)
        # normalise row vectors
        R /= sum(R, 1).reshape(R.shape[0], 1)

    def update_gating(self, Xf):
        """Updates the gating weight vectors and priors until convergence,
        based on the current classifier models. Convergence is determined
        by monitoring the variational bound of the gating network. The method
        returns (iterations, L), where iterations is the number of iterations
        that are performed until convergence, and L is the variational
        bound after the last iteration. If the maximum number of iterations
        was exceeded, then (-1, L) is returned.
        """
        N, Dv = Xf.shape
        K = len(self.cls)
        # local caches
        b_a, b_b = Gating.b_a, Gating.b_b
        L_b_const = Gating.L_b_const
        b = self.b
        max_dL, max_iter = Gating.max_dL, Gating.max_iter
        # pre-update b_a, as it stays the same in each iteration
        b[:,0] = Gating.b_a + 0.5 * float(Dv)
        gammaln_b_a = sum(gammaln(b[:,0]))

        # start iteration
        i, dL, L = 0, max_dL + 1.0, -inf
        while (dL > max_dL) and (i < max_iter):
            # update responsibilities
            self.update_resp(Xf)
            # update priors (only b_b, as b_a was updated before)
            V = self.V
            b[:,1] = b_b + 0.5 * (sum(V * V, 0) + self.cov_Tr)
            # update gating weights
            id, KL = self.update_weights(Xf)
            if id < 0:
                #print "Iteration limit exceeded when updating gating weights"
                pass
            # get new variational bound
            L_prev = L
            # E_V_b = E(ln p(V | b)) + E(ln p(b)) - E(ln q(b)) - E(ln q(V))
            E_V_b = K * L_b_const + gammaln_b_a - sum(b[:,0] * log(b[:,1])) + \
                    0.5 * self.ln_cov_det + 0.5 * K * Dv
            L = E_V_b + KL
            #print L, exp(self.V)
            # as we are using an approximation, the variational bound
            # might decrease, so we're not checking and need to take the abs()
            dL = abs(L - L_prev)
            # next iteration
            i += 1

        if dL > max_dL:
            return (-1, L)
        else:
            return (i, L)

    def var_bound(self, Xf):
        """Returns the variational bound of classifiers and gating network.
        """
        cls, b, R = self.cls, self.b, self.R
        N, Dv = Xf.shape
        K = len(cls)
        # get the classifier variational bound
        cl_var_bound = 0.0
        for k in xrange(K):
            cl_var_bound += cls[k].var_bound(R[:,k])
        # get the gating network variational bound
        # E(ln p(V | b)) + E(ln p(b)) - E(ln q(b)) - E(ln q(V))
        E_V_b = K * Gating.L_b_const + \
                sum(gammaln(b[:,0])) - sum(b[:,0] * log(b[:,1])) + \
                0.5 * self.ln_cov_det + 0.5 * K * Dv
        # E(ln p(Z | V)) - E(ln q(Z))
        E_Z = KL = sum(R * nan_to_num(log(self.gating_matrix(Xf) / R)))
        return cl_var_bound + E_V_b + E_Z

    def ln_model_prob(self, Xf):
        """Returns the ln of the model probability, which is the
        variational bound / K! to account for symmetries in the classifier
        matching function.
        """
        # we need to use the log function from the math module,
        # as the one from the numpy module cannot handle numbers of type 'long'
        return self.var_bound(Xf) - math_log(fact(len(self.cls)))

    def gating_weights(self, xf, x):
        """Returns the gating vector for the given input x and gating
        features xf.
        """
        # for detailed comments see gating_matrix()
        cls = self.cls
        K = len(cls)
        g = dot(xf, self.V)
        g = exp(clip(g, exp_min, ln_double_max - log(K)))
        for k in xrange(K):
            g[k] *= cls[k].matches(x)
        g /= sum(g)
        g[isnan(g)] = 1.0 / K
        return g

    def pred(self, x, xf):
        """Returns the prediction mean for a new input x with gating
        features xf.
        """
        g = self.gating_weights(xf, x)
        means = vstack([cl.pred(x) for cl in self.cls])
        return dot(g, means)

    def pred_var(self, x, xf):
        """Returns the prediction mean and variance for a new input x with
        gating feature xf. The return value is (means, variances).
        """
        g = self.gating_weights(xf, x)
        # get means and variances
        means_vars = [cl.pred_var(x) for cl in self.cls]
        means = vstack([m_v[0] for m_v in means_vars])
        variances = array([m_v[1] for m_v in means_vars], double)
        # get mean and variance vector
        mean = dot(g, means)
        var = dot(g, variances.reshape(variances.shape[0], 1) +
                     power(means, 2.0)) - power(mean, 2.0)
        return (mean, var)
