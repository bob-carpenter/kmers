# Sample code automatically generated on 2021-12-11 17:57:52
# by www.matrixcalculus.org from input: 
#     d/dtheta y' * log(x * theta) - 1/18 * theta' * theta
#       = x' * (y ./ (x * theta)) - 2/18 * theta
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
        y ~ multinomial(x * theta)
        theta ~ normal(0, 3)
    where
        * y: M x 1 array of integer counts
        * x: M x T left stochastic matrix of kmer probabilities for isoform
        * theta: T x 1 simplex of isoform expression
    with size constants
        * K: size of k-mer
        * M = 4^K: number of distinct k-mers
        * T: number of target isoforms

    All operations are on the log scale, with target log posterior
        log p(theta | y) = y' * log(x * theta) - 1/18 * theta' * theta

    The log posterior could be considered a penalized maximum likelihood with
    a scaled L2 penalty 1/18 * ||theta||_2^2

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
        t_0 = (x).dot(theta)
        functionValue = ((y).dot(np.log(t_0)) - ((theta).dot(theta) / 18))
        gradient = ((x.T).dot((y / t_0)) - ((2 / 18) * theta))
        return functionValue, gradient
