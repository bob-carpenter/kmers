import kmerexpr.multinomial_simplex_model as mm
import kmerexpr.transcriptome_reader as tr
import numpy as np
import os
import pytest
from scipy import optimize

from test.conftest import check_gradient
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def test1():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.multinomial_simplex_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == 3
    assert model.M() == 16
    alpha = 0.5 * np.ones(model.T())
    theta_test = np.random.dirichlet(alpha)
    check_gradient(model, theta_test)

@pytest.mark.slow
def test_human_transcriptome():
    ISO_FILE = os.path.join(ROOT, "data", "GRCh38_latest_rna.fna")
    X_FILE = os.path.join(DATA_PATH, "xgrch38_csr.npz")
    K = 3  # make = 10 for full test
    M = 4 ** K
    T = 81456  # Rob: It's coming out at 81456 in my test. It was 80791 before
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    print("finished writing x to file")
    y_test = np.random.poisson(20, 4 ** K)
    print("reading in model")
    model = mm.multinomial_simplex_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == T
    assert model.M() == M
    alpha = 0.1 * np.ones(model.T())
    theta_test = np.random.dirichlet(alpha)
    print("checking gradient")
    check_gradient(model, theta_test)


def test_optimizer():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    print(X_FILE)
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.multinomial_simplex_model(X_FILE, y_test)
    os.remove(X_FILE)
    theta0 = np.random.dirichlet(0.7 * np.ones(model.T()))
    # Get high precision solution
    dict_sol = model.fit(theta0)
    # Compare against something scipy optimizer?
    # cons = (
    #     {"type": "ineq", "fun": lambda x: x[0] - 2 * x[1] + 2},
    #     {"type": "ineq", "fun": lambda x: -x[0] - 2 * x[1] + 6},
    #     {"type": "ineq", "fun": lambda x: -x[0] + 2 * x[1] + 2},
    # )
    # bnds = (0, 1)
    # theta_sol, f_sol, dict_sol = optimize.fmin_l_bfgs_b(func, theta, fprime)
    # assert np.linalg.norm(CGResult.x - theta_sol) < 1e-04
    # assert np.linalg.norm(CGResult.jac - dict_sol["grad"]) < 1e-04
    # assert np.linalg.norm(dict_sol["grad"]) < 1e-04
