import kmerexpr.normal_model as mm
import kmerexpr.transcriptome_reader as tr
import numpy as np
from scipy import optimize
import os
import pytest
from scipy.linalg import solve
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")



def test_least_squares_solution():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    print(X_FILE)
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.normal_model(X_FILE, y_test)
    os.remove(X_FILE)
    # Get high precision solution
    dict_sol = model.fit(n_iters = 100)
    
    X_array = model.x.toarray()
    Xt= X_array.transpose()
    A = Xt@X_array
    b_right = Xt@y_test/y_test.sum()
    # Test leasts squares solution is close to solving linear system
    assert np.linalg.norm(A@dict_sol["x"] - b_right) < 1e-02
    ## Currently unable to get a comparison with lbfgs with box bounds working. I suspect that the code no longer works with bounds
    # func = lambda theta: np.linalg.norm(X_array@theta - y_test, ord=2)**2
    # fprime = lambda theta: 2*(A*theta-b_right)
    # theta0 = np.random.dirichlet(0.7 * np.ones(model.T()))
    # import pdb; pdb.set_trace()
    # lower = np.zeros(model.T())
    # upper = np.ones(model.T())
    # lbfgs_bounds = zip(lower.tolist(), upper.tolist())
    # lbfgs_bounds = list(lbfgs_bounds)
    # theta_sol, f_sol, dict_flags_convergence = optimize.fmin_l_bfgs_b(func, theta0 , fprime=fprime , bounds = lbfgs_bounds)  # pgtol = gtol, factr = factr, maxiter=n_iters, maxfun = 10*n_iters

