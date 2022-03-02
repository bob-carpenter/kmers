# Sample code automatically generated on 2021-12-11 17:57:52
# by www.matrixcalculus.org from input:
#     d/dtheta y' * log(x * exp(theta) / sum(exp(theta))) - 1/18 * theta' * theta
#       = 1 / sum(exp(theta))
#         * (x' * (y ./ (1 / sum(exp(theta)) * x * exp(theta))))
#         .* exp(theta)
#       - 1 / sum(exp(theta)).^2
#         * exp(theta)' * x'
#         * (y ./ (1 / sum(exp(theta)) * x * exp(theta)))
#         * exp(theta)
#       - 2/18 * theta
# where
#     theta is a vector
#     x is a matrix
#     y is a vector
# The generated code is provided "as is" without warranty of any kind.

# The code here refactors the auto-generated code into a class and
# pulls the testing out.

import numpy as np
from scipy.sparse import load_npz
from scipy import optimize
from exp_grad_solver import exp_grad_solver
from scipy.special import softmax as softmax
# BMW: Class names are usually done in CamelCase style
class multinomial_simplex_model:
    """Multinomial model of k-mer reads with a simplex constraint

    The likelihood and prior are
        y ~ multinomial(x * theta)
        theta ~ dirichlet(beta)
    where
        * y: M x 1 array of integer counts
        * x: M x T left stochastic matrix of kmer probabilities for isoform
        * theta: T x 1 vector of expression values
    with size constants
        * K: size of k-mer
        * M = 4^K: number of distinct k-mers
        * T: number of target isoforms

    All operations are on the log scale, with target log posterior
        log p(theta | y) = y' * log(x * theta) - (beta-1) * sum(theta), subject to theta in T-simplex

    The constructor instantiates a model based on two arguments
    corresponding to x and y.  Because x is so large, it is loaded from
    a file to which it has been serialized in csr .npz format, which is
    described in the scipy docs
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html

    :param x_file: file in csr .npz format from scipy.sparse
    containing the x matrix

    :param y: vector of read counts
    """

    def __init__(self, x_file=None, y=None, beta=0.5):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y -- vector of k-mer counts
        N -- total number of k-mers
        """
        self.x = load_npz(x_file)
        self.y = y
        self.N = np.sum(y)
        self.beta = beta

    def M(self):
        return self.x.shape[0]

    def T(self):
        return self.x.shape[1]

    def logp_grad(self, theta=None, eps = 10**(-10)):
        """Return negative log density and its gradient evaluated at the
        specified simplex.
         loss(theta) = y' log(X'theta) + (beta-1 )
         grad(theta) = X y diag{X' theta}^{-1}+ (beta-1 )
        Keyword arguments:
        theta -- simplex of expected isoform proportions
        """
        x = self.x
        y = self.y
        dim = theta.shape
        assert len(dim) == 1
        theta_rows = dim[0]
        dim = x.shape
        assert len(dim) == 2
        x_rows = dim[0]
        x_cols = dim[1]
        dim = y.shape
        assert len(dim) == 1
        y_rows = dim[0]
        assert y_rows == x_rows
        assert theta_rows == x_cols
        # import pdb; pdb.set_trace()
        xtheta = x.dot(theta) #softmax(x.dot(theta) +eps)
        ymask = y.nonzero()
        ynnz = y[ymask]  # Need only consider coordinates where y is nonzero
        xthetannz = xtheta[ymask]
        functionValue = ynnz.dot(np.log(xthetannz)) + self.beta - 1
        yxTtheta = ynnz / xthetannz
        t1 = x[ymask].T.dot(yxTtheta)
        gradient = t1 + self.beta - 1
        return functionValue, gradient

    def fit(self, theta=None, tol=10.0**(-4), gtol=1e-10, n_iters = 10000, lrs =0.1*np.ones(1)):
        # loss_grad = lambda x: self.logp_grad(x)
        dict_sol = exp_grad_solver(self.logp_grad, theta, lrs =lrs, tol = tol, gtol=gtol, n_iters = n_iters)
        # cons = (
        #     {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
        #     {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
        #     {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
        # )
        # bnds = (0, 1)
        # theta_sol, f_sol, dict_sol = optimize.fmin_l_bfgs_b(func, theta, fprime)
        f_sol = dict_sol['loss_records'][-1]
        return dict_sol['x'], f_sol, dict_sol  #{'x' : x, 'norm_records' : norm_records, 'loss_records' : loss_records, 'time_records' : time_records }
