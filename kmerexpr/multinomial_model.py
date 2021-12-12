# Sample code automatically generated on 2021-12-11 17:57:52
# by www.matrixcalculus.org from input: 
#     d/dtheta y' * log(x * theta) = x'*(y./(x*theta))
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

    The model is y ~ multinomial(x * theta), with variables

    * y: M x 1 array of integer counts
    * x: M x T left stochastic matrix of kmer probabilities for isoform
    * theta: T x 1 simplex of isoform expression

    and sizes are given by

    * K: size of k-mer
    * M = 4^K: number of distinct k-mers
    * T: number of target isoforms

    The log density All operations are the log scale.

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

    def logp_grad(theta = None):
        """Return log density and its gradient evaluated for the
        specified simplex of isoform proportions.
        
        Keyword arguments:
        theta -- simplex of expected isoform proportions
        """
        assert isinstance(theta, np.ndarray)
        dim = theta.shape
        assert len(dim) == 1
        theta_rows = dim[0]
        assert isinstance(self.x, np.ndarray)
        dim = self.x.shape
        assert len(dim) == 2
        x_rows = dim[0]
        x_cols = dim[1]
        assert isinstance(self.y, np.ndarray)
        dim = self.y.shape
        assert len(dim) == 1
        y_rows = dim[0]
        assert y_rows == x_rows
        assert theta_rows == x_cols

        phi = (self.x).dot(theta)
        functionValue = (self.y).dot(np.log(phi))
        gradient = (self.x.T).dot((self.y / phi))
        return functionValue, gradient

def checkGradient(model = None, theta = None):
    """Test gradient of model at specified argument using finite differences.

    Test is based on error in finite difference gradient

    f(x + t * delta) - f(x - t * delta) / (2t)

    compared to defined gradient

    <g, delta> = g' * delta

    The test generates a standard normal delta to use for testing and
    then prints the Euclidean distance between the results, i.e., the
    norm of the finite-difference approximation error.

    The model needs to implement a logp_grad(theta) method that returns
    a pair consisting of the log density and its gradient evaluated at
    the specified simplex.

    Keyword arguments:
    model -- instance of multinomial_model to evaluate
    theta -- simplex of expected isoform proportions at which to
    evaluate
    """
    t = 1E-6
    delta = np.random.randn(3)
    f1, _ = model.logp_grad(theta + t * delta)
    f2, _ = model.logp_grad(theta - t * delta)
    f, g = model.logp_grad(theta)
    err = (f1 - f2) / (2 * t) - np.tensordot(g, delta, axes=1)
    print('norm of approximation error', np.linalg.norm(err))


if __name__ == '__main__':
    theta = np.random.randn(3)
    x = np.random.randn(3, 3)
    y = np.random.randn(3)
    model = multinomial_model(
    functionValue, gradient = fAndG(theta, x, y)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(theta, x, y)        
