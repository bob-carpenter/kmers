import numpy as np
from scipy.sparse import load_npz
from scipy.special import softmax as softmax
from scipy import optimize
import time
from scipy.sparse.linalg import lsqr

# BMW: Class names are usually done in CamelCase style
class normal_model:
    """Normal model of k-mer reads.

    The likelihood and prior are
        y ~ normal(x * theta)
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
        log p(theta | y) =  -||y - sum(y)*x * psi ||^2 - beta * psi' * psi

    The constructor instantiates a model based on two arguments
    corresponding to x and y.  Because x is so large, it is loaded from
    a file to which it has been serialized in csr .npz format, which is
    described in the scipy docs
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.save_npz.html

    :param x_file: file in csr .npz format from scipy.sparse
    containing the x matrix

    :param y: vector of read counts
    """

    def __init__(self, x_file=None, y_file=None, beta =0.0001, lengths=None, solver='lsq_linear'):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y -- vector of k-mer counts
        N -- total number of k-mers
        """
        self.x = load_npz(x_file)
        if(isinstance(y_file, np.ndarray)):
            self.y = y_file
        else:
            self.y = np.load(y_file)
        self.N = np.sum(self.y)
        self.beta = beta
        self.name = "normal"
        self.solver ="lsq_linear"
    def M(self):
        return self.x.shape[0]

    def T(self):
        return self.x.shape[1]


    def fit(self, model_parameters, theta0=None, tol=1e-10, n_iters = 10000):

        start = time.time()
        # x, istop, itn = lsqr(self.x, self.y/self.N, damp=self.beta, x0=x0)[:3]
        results = optimize.lsq_linear(self.x, self.y/self.N, bounds=(0, 1), tol=tol, max_iter =  n_iters )
        end = time.time()
        results['x'] = results['x']/results['x'].sum()   #normalizing back onto simplex
 
        print("normal model took ", end - start, " time to fit")
        # dict_opt = {'x' : x, 'iteration_counts' : itn} 
        return results
