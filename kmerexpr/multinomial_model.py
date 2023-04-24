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
from scipy.special import softmax as softmax
from scipy import optimize
import time
from kmerexpr.rna_seq_reader import load_xy


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
        penalty(theta) = beta * ||theta||_2^2
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

    def __init__(
        self, x_file=None, y_file=None, beta=1 / 18, lengths=None, solver_name="lbfgs"
    ):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y -- vector of k-mer counts
        beta -- parameter for prior
        """
        x, y = load_xy(x_file, y_file)
        self.ymask = (
            y.nonzero()
        )  # Need only need self.ynnz and self.xnnz. Throw away the rest?
        self.ynnz = y[self.ymask]
        self.xnnz = x[self.ymask]
        self.N = np.sum(y)
        self.name = "softmax"
        x_dim = x.shape
        self.M = x_dim[0]
        self.T = x_dim[1]
        self.beta = beta
        self.solver_name = solver_name
        # dimension checking
        assert len(x_dim) == 2
        x_rows = x_dim[0]
        x_cols = x_dim[1]
        dim = y.shape
        assert len(dim) == 1
        y_rows = dim[0]
        assert y_rows == x_rows

    def logp_grad_old(self, theta=None):
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

        ymask = y.nonzero()
        ynnz = y[ymask]
        sig = softmax(theta)
        xTsig = x.dot(sig)
        xTsignnz = xTsig[ymask]
        t_3 = (x[ymask].T).dot(ynnz / xTsignnz)
        functionValue = ynnz.dot(np.log(xTsignnz)) - (theta.dot(theta) * self.beta)
        gradient = t_3 * sig - self.N * sig - (2 * self.beta) * theta
        # Double check: Think ((sig).dot(t_3)*sig )) = sum(y)*sig = N*sig
        return functionValue, gradient

    def logp_grad(self, theta=None):
        """Return log density and its gradient evaluated at the
        specified simplex.

        Keyword arguments:
        theta -- simplex of expected isoform proportions
        """
        sig = softmax(theta)
        xTsignnz = self.xnnz.dot(sig)
        t_3 = (self.xnnz.T).dot(self.ynnz / xTsignnz)
        functionValue = self.ynnz.dot(np.log(xTsignnz)) - (theta.dot(theta) * self.beta)
        gradient = t_3 * sig - self.N * sig - (2 * self.beta) * theta
        return functionValue, gradient

    def fit(
        self,
        model_parameters,
        theta0=None,
        factr=1.0,
        gtol=1e-12,
        tol=None,
        n_iters=50000,
    ):
        if theta0 is None:  # initialize to normal 0 1
            theta0 = np.random.normal(0, 1, self.T)

        func = lambda theta: -self.logp_grad(theta)[0]
        fprime = lambda theta: -self.logp_grad(theta)[1]
        start = time.time()
        theta_sol, f_sol, dict_flags_convergence = optimize.fmin_l_bfgs_b(
            func,
            theta0,
            fprime,
            pgtol=gtol,
            factr=factr,
            maxiter=n_iters,
            maxfun=10 * n_iters,
        )
        end = time.time()
        print("softmax model took ", end - start, " time to fit")
        if dict_flags_convergence["warnflag"] == 1:
            print(
                "WARNING: softmax model did not converge. too many function evaluations or too many iterations. Print d[task]:",
                dict_flags_convergence["task"],
            )
            print("Total iterations: ", str(dict_flags_convergence["nit"]))
        elif dict_flags_convergence["warnflag"] == 2:
            print(
                "WARNING: softmax model did not converge due to: ",
                dict_flags_convergence["task"],
            )
        dict_opt = {
            "x": softmax(theta_sol),
            "loss_records": -f_sol,
            "iteration_counts": dict_flags_convergence["nit"],
            "grad": -dict_flags_convergence["grad"],
        }
        return dict_opt
