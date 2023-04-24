import kmerexpr.multinomial_simplex_model as mm
import kmerexpr.transcriptome_reader as tr
from kmerexpr.utils import load_lengths, Problem, Model_Parameters
import numpy as np
import os
import pytest
from scipy import optimize

from test.conftest import check_gradient

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def test_grad_setup():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4**K)
    model = mm.multinomial_simplex_model(X_FILE, y_test)
    assert model.T == 3
    assert model.M == 16
    alpha = 0.5 * np.ones(model.T)
    theta_test = np.random.dirichlet(alpha)
    check_gradient(model, theta_test)
    model = mm.multinomial_simplex_model(X_FILE, y_test, beta=0.1)
    check_gradient(model, theta_test)
    os.remove(X_FILE)


@pytest.mark.slow
def test_human_transcriptome():
    ISO_FILE = os.path.join(ROOT, "data", "GRCh38_latest_rna.fna")
    X_FILE = os.path.join(DATA_PATH, "xgrch38_csr.npz")
    K = 3  # make = 10 for full test
    M = 4**K
    T = 81456  # Rob: It's coming out at 81456 in my test. It was 80791 before
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    print("finished writing x to file")
    y_test = np.random.poisson(20, 4**K)
    print("reading in model")
    model = mm.multinomial_simplex_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T == T
    assert model.M == M
    alpha = 0.1 * np.ones(model.T)
    theta_test = np.random.dirichlet(alpha)
    print("checking gradient")
    check_gradient(model, theta_test)


def test_optimizer():
    filename = "test4.fsa"
    ISO_FILE = os.path.join(DATA_PATH, filename)
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    print(X_FILE)
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4**K)
    model_parameters = Model_Parameters(
        model_type="simplex", solver_name="exp_grad", lrs="warmstart"
    )
    model = model_parameters.initialize_model(X_FILE, y_test)
    dict_sol = model.fit(model_parameters, n_iters=500)
    # Compare against softmax+lbfgs solution
    import kmerexpr.multinomial_model as mm_softmax

    model_softmax = mm_softmax.multinomial_model(X_FILE, y_test)
    theta0_softmax = np.random.normal(0, 1, model.T)
    dict_sol_softmax = model_softmax.fit(
        theta0_softmax, factr=0.0001, gtol=1e-17, n_iters=500
    )
    np.testing.assert_allclose(
        model_softmax.logp_grad(dict_sol["x"])[0],
        model_softmax.logp_grad(dict_sol_softmax["x"])[0],
        rtol=1e-3,
        atol=0.3,
    )
    os.remove(X_FILE)