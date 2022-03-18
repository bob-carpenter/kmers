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
from scipy.special import softmax as softmax
from scipy import optimize
import time

# BMW: Class names are usually done in CamelCase style
class multinomial_model:
    """Multinomial model of k-mer reads.

    The likelihood and prior are
        y ~ multinomial(x * softmax(theta))
        theta ~ normal(0, 3)
    where
        * y: M x 1 array of integer counts
        * x: M x T left stochastic matrix of kmer probabilities for isoform
        * theta: T x 1 vector of expression values
    with size constants
        * K: size of k-mer
        * M = 4^K: number of distinct k-mers
        * T: number of target isoforms

    All operations are on the log scale, with target log posterior
        log p(theta | y) = y' * log(x * softmax(theta)) - 1/18 * theta' * theta
    where
        softmax(theta) = exp(theta) / sum(exp(theta))

    Because theta is T x 1, the likelihood itself is not identified as
    theta + c yields the same density as theta for any constant c.  The
    parameters are identified through the prior/penalty.

    The log posterior could be considered a penalized maximum likelihood with
    a scaled L2 penalty
        penalty(theta) = 1/18 * ||theta||_2^2
    The penalty will shrink estimates of theta toward zero, which has he effect
    of making softmax(theta) more uniform.

    The constructor instantiates a model based on two arguments
    corresponding to x and y.  Because x is so large, it is loaded from
    a file to which it has been serialized in csr .npz format, which is
    described in the scipy docs
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html

    :param x_file: file in csr .npz format from scipy.sparse
    containing the x matrix

    :param y: vector of read counts
    """

    def __init__(self, x_file=None, y_file=None):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y -- vector of k-mer counts
        N -- total number of k-mers
        """
        self.x = load_npz(x_file)
        self.y = np.load(y_file)
        self.N = np.sum(self.y)
        self.name = "softmax+lbfgs"

    def M(self):
        return self.x.shape[0]

    def T(self):
        return self.x.shape[1]

    def logp_grad(self, theta=None):
        """Return log density and its gradient evaluated at the
        specified simplex.

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
        ymask = y.nonzero()
        ynnz = y[ymask] 
        sig = softmax(theta)
        xTsig = x.dot(sig)
        xTsignnz = xTsig[ymask]
        t_3 = (x[ymask].T).dot(ynnz / xTsignnz)
        functionValue = ynnz.dot(np.log(xTsignnz)) - (theta.dot(theta) / 18)
        gradient = t_3 * sig - self.N * sig - (2 / 18) * theta
        # Double check: Think ((sig).dot(t_3)*sig )) = sum(y)*sig = N*sig
        return functionValue, gradient

    def fit(self, theta=None, factr=10.0, gtol=1e-10, n_iters = 10000):
        func = lambda theta: -self.logp_grad(theta)[0]
        fprime = lambda theta: -self.logp_grad(theta)[1]
        start = time.time()
        theta_sol, f_sol, dict_sol = optimize.fmin_l_bfgs_b(func, theta, fprime, pgtol = gtol, factr = factr, maxiter=n_iters)
        end = time.time()
        print("softmax model took ", end - start, " time to fit")
        dict_sol["grad"] = -dict_sol["grad"]
        return softmax(theta_sol), -f_sol, dict_sol
