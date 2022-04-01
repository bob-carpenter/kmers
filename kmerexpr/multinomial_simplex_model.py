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

    def __init__(self, x_file=None, y_file=None, beta=1.0):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y_file -- path to total counts of each kmer
        beta -- the parameter of the Dirchlet distribution. Default beta =1 is equivalent to using no prior or a uniform prior
        y -- vector of k-mer counts
        """
        self.x = load_npz(x_file)
        if(isinstance(y_file, np.ndarray)):
            self.y = y_file
        else:
            self.y = np.load(y_file)
        self.N = np.sum(self.y)
        self.beta = beta
        self.name = "mirror"

    def M(self):
        return self.x.shape[0]

    def T(self):
        return self.x.shape[1]

    def logp_grad_(self, theta, x, y):
        xtheta = x.dot(theta) #softmax(x.dot(theta) +eps)
        ymask = y.nonzero()
        ynnz = y[ymask]  # Need only consider coordinates where y is nonzero
        xthetannz = xtheta[ymask]
        functionValue = ynnz.dot(np.log(xthetannz)) + (self.beta - 1)*np.sum(np.log(theta))
        yxTtheta = ynnz / xthetannz
        t1 = x[ymask].T.dot(yxTtheta)
        gradient = t1 + (self.beta - 1)/theta
        return functionValue, gradient

    def logp_grad(self, theta = None, eps = 10**(-10), batch = None):
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
        if isinstance(batch,(np.ndarray)):
            return self.logp_grad_(theta, x[batch], y[batch] )
        else:
            return self.logp_grad_(theta, x, y )
            


    def fit(self, theta0=None, tol=1e-8, gtol=1e-8, n_iters = 100, lrs = None,  batchsize = None, continue_from =0):

        if theta0 is None:  #initialize to uniform
            alpha = np.ones(self.T())
            theta0 = alpha/alpha.sum()

        if batchsize is None:
            batchsize = int(self.M()/5)
        elif batchsize == "full":
            batchsize = None

        dict_sol = exp_grad_solver(self.logp_grad, theta0, lrs =lrs, tol = tol, gtol=gtol, n_iters = n_iters,  batchsize = batchsize, n = self.M(), continue_from = continue_from)
        return dict_sol