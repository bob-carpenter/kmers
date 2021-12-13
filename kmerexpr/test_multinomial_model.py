import multinomial_model as mm
import transcriptome_reader as tr
import numpy as np
import os
import pytest
from scipy.sparse import load_npz
from scipy.special import softmax

def check_gradient(model = None, theta = None, tol = 1e-3):
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
    ISO_FILE ='kmerexpr/test_data/test4.fsa'
    X_FILE = "kmerexpr/test_data/x4_csr.npz"
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4**K)
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == 3
    assert model.M() == 16
    theta_test = softmax(np.random.normal(0, 1, model.T()))
    check_gradient(model, theta_test)

def test_human_transcriptome():
    ISO_FILE ='data/GRCh38_latest_rna.fna'
    X_FILE = 'kmerexpr/test_data/xgrch38_csr.npz'
    K = 10
    M = 4**K
    T = 80791
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    print("finished writing x to file")
    y_test = np.random.poisson(20, 4**K)
    print("reading in model")
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T() == T
    assert model.M() == M
    theta_test = softmax(np.random.normal(0, 1, T))
    print("checking gradient")
    check_gradient(model, theta_test)

    
    
    
    
