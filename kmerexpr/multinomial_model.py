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
    
    def __init__(self, x_file = None, y = None):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y -- vector of k-mer counts
        """
        self.x = load_npz(x_file)
        self.y = y

    def M(self):
        return self.x.shape[0]

    def T(self):
        return self.x.shape[1]
    
    def logp_grad(self, theta = None):
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

        t_0 = np.exp(theta)
        t_1 = ((1 / np.sum(t_0)) * (x).dot(t_0))
        t_2 = np.sum(t_0)
        t_3 = (x.T).dot((y / t_1))
        functionValue = ((y).dot(np.log(t_1)) - ((theta).dot(theta) / 18))
        gradient = ((((1 / t_2) * (t_3 * t_0)) - ((1 / (t_2 ** 2)) * ((t_0).dot(t_3) * t_0))) - ((2 / 18) * theta))

        return functionValue, gradient
