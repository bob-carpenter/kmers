import kmerexpr.multinomial_model as mm
import kmerexpr.transcriptome_reader as tr
import numpy as np
import os
import pytest
from scipy import optimize
from test.conftest import check_gradient
from scipy.special import softmax as softmax

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def test1():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    X_FILE = os.path.join(DATA_PATH, "x4_csr.npz")
    K = 2
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
    y_test = np.random.poisson(5, 4 ** K)
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T == 3
    assert model.M == 16
    theta_test = np.random.normal(0, 1, model.T)
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
    model = mm.multinomial_model(X_FILE, y_test)
    os.remove(X_FILE)
    assert model.T == T
    assert model.M == M
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
    theta0 = np.random.normal(0, 1, model.T)
    # Get high precision solution
    dict_sol = model.fit(theta0, factr=1.0, gtol=1e-16)
    # Compare against CG
    func = lambda theta: -model.logp_grad(theta)[0]
    fprime = lambda theta: -model.logp_grad(theta)[1]
    CGResult = optimize.minimize(
        func, theta0, method="CG", jac=fprime, options={"gtol": 1e-16}
    )
    assert np.linalg.norm(softmax(CGResult.x) - dict_sol["x"])/np.linalg.norm(dict_sol["x"]) < 1e-04
    assert np.linalg.norm(CGResult.jac - dict_sol["grad"]) < 1e-03
    assert np.linalg.norm(dict_sol["grad"]) < 1e-03
