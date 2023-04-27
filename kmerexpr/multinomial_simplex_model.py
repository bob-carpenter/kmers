import numpy as np
from scipy.sparse.linalg import lsqr
from kmerexpr.exp_grad_solver import exp_grad_solver
from kmerexpr.mg import mg
from kmerexpr.transcriptome_reader import load_xy
import kmerexpr.utils as utils


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
        log p(theta | y) = y' * log(x * theta) - (beta-1) * ( sum(log theta) - log sum(theta / Lengths) )),

    subject to theta in T-simplex, where Lengths is the T-vector of lengths of the reference set of isoforms.

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
        self, x_file=None, y_file=None, beta=1.0, lengths=None, solver_name="exp_grad"
    ):
        """Construct a multinomial model.

        Keyword arguments:
        x_file -- path to file containing a serialized sparse,
        left-stochastic matrix in .npz format from scipy.sparse
        y_file -- path to total counts of each kmer
        beta -- the parameter of the Dirchlet distribution. Default beta =1 is equivalent to using no prior or a uniform prior
        y -- vector of k-mer counts
        lengths -- an array of the lengths of the isoforms
        solver_name -- a string that is either mirror_bfgs or exp_grad, which are the two available solvers for fitting the model
        """
        x, y = load_xy(x_file, y_file)
        self.ymask = (
            y.nonzero()
        )  # Need only need self.ynnz and self.xnnz. Throw away the rest?
        self.ynnz = y[self.ymask]
        self.xnnz = x[self.ymask]
        self.qnnz = self.ynnz / np.sum(self.ynnz)  #normalize to probabilities
        self.N = np.sum(y)
        self.beta = beta
        self.name = "mirror"
        x_dim = x.shape
        self.M = x_dim[0]
        self.T = x_dim[1]
        self.lengths = lengths
        self.solver_name = solver_name
        # dimension checking
        assert len(x_dim) == 2
        x_rows = x_dim[0]
        x_cols = x_dim[1]
        dim = y.shape
        assert len(dim) == 1
        y_rows = dim[0]
        assert y_rows == x_rows
        if lengths is not None:
            assert len(lengths) == x_cols
        else:
            self.lengths = np.ones(x_cols)
        self.scratch = np.zeros(self.xnnz.shape[0], dtype=self.xnnz.dtype)

    def logp_grad(self, theta, nograd=False):
        """Return negative log density and its gradient evaluated at the specified simplex.

           loss(theta) = y' log(X'theta) + (beta-1 )(sum(log(theta)) - log sum (theta/Lenghts))
           grad(theta) = X y diag{X' theta}^{-1}+ (beta-1 ) (1 - (1/Lenghts)/sum (theta/Lenghts) )
        Keyword arguments:
        theta -- simplex of expected isoform proportions
        """
        return utils.logp_grad(theta, self.beta, self.xnnz, self.qnnz, self.scratch, self.lengths, nograd=nograd)

    def initialize_iterates_uniform(self, lengths=None):
        # should use beta and dichlet to initialize? Instead of always uniform?
        alpha = np.ones(self.T)
        theta0 = alpha / alpha.sum()
        return theta0

    def initialize_iterates_Xy(self):
        theta0 = self.ynnz @ self.xnnz
        theta0 = theta0 / theta0.sum()
        return theta0

    def initialize_iterates_lsq(self, iterations=200, mult_fact_neg=10):
        theta0, istop, itn, r1norm = lsqr(
            self.xnnz, self.ynnz / self.N, iter_lim=iterations
        )[:4]
        mintheta = np.min(theta0[theta0 > 0])
        theta0[theta0 <= 0] = mintheta / mult_fact_neg
        theta0 = theta0 / theta0.sum()
        return theta0

    def fit(
        self,
        model_parameters,
        theta0=None,
        tol=1e-20,
        gtol=1e-20,
        n_iters=100,
        opt_method="mg",
    ):
        if theta0 is None:  # initialize to uniform
            if model_parameters.init_iterates == "lsq":
                theta0 = self.initialize_iterates_lsq()
            elif model_parameters.init_iterates == "Xy":
                theta0 = self.initialize_iterates_Xy()
            else:
                theta0 = self.initialize_iterates_uniform()

        if opt_method == "mg":
            dict_sol = mg(self.logp_grad, theta0, tol=tol, max_iter=n_iters)
        else:
            dict_sol = exp_grad_solver(
                self.logp_grad,
                theta0,
                lrs=model_parameters.lrs,
                tol=tol,
                gtol=gtol,
                n_iters=n_iters,
            )

        return dict_sol
