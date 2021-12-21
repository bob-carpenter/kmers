import kmerexpr.multinomial_model as mm
import kmerexpr.transcriptome_reader as tr
import numpy as np
import os
import pytest
from scipy import optimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def check_gradient(model=None, theta=None, tol=1e-3):
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

    The only assumption on the density class here is that it operates on
    unconstrained parameter values.

    Keyword arguments:
    model -- instance of multinomial_model defining log density
    theta -- simplex of parameters for isoform proportions
    """
    t = 1e-6
    T = theta.shape[0]
    delta = np.random.randn(T)
    f1, _ = model.logp_grad(theta + t * delta)
    f2, _ = model.logp_grad(theta - t * delta)
    f, g = model.logp_grad(theta)
    deriv_finite_diff = (f1 - f2) / (2 * t)
    deriv_fun = np.tensordot(g, delta, axes=1)
    err = deriv_finite_diff - deriv_fun
    assert 0 == pytest.approx(0, tol)


def test1():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == 3
    assert model.M() == 16
    theta_test = np.random.normal(0, 1, model.T())
    check_gradient(model, theta_test)


def test_human_transcriptome():
    ISO_FILE = os.path.join(ROOT, "data", "GRCh38_latest_rna.fna")
    X_FILE = os.path.join(DATA_PATH, "xgrch38_csr.npz")
    K = 3  # make = 10 for full test
    M = 4 ** K
    T = 80791  # Rob: It's coming out at 81456 in my test. Something wrong?
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    print("finished writing x to file")
    y_test = np.random.poisson(20, 4 ** K)
    print("reading in model")
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == T
    assert model.M() == M
    theta_test = np.random.normal(0, 1, T)
    print("checking gradient")
    check_gradient(model, theta_test)


def test_optimizer():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    print(X_FILE)
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    theta0 = np.random.normal(0, 1, model.T())
    # Get high precision solution
    theta_sol, f_sol, dict_sol = model.fit(theta0, factr=1.0, pgtol=1e-14)
    # Compare against CG
    func = lambda theta: -model.logp_grad(theta)[0]
    fprime = lambda theta: -model.logp_grad(theta)[1]
    CGResult = optimize.minimize(
        func, theta0, method="CG", jac=fprime, options={"gtol": 1e-14}
    )
    assert np.linalg.norm(CGResult.x - theta_sol) < 1e-04
    assert np.linalg.norm(CGResult.jac - dict_sol["grad"]) < 1e-04
    assert np.linalg.norm(dict_sol["grad"]) < 1e-04
