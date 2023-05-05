import numpy as np
import os
import pytest
from scipy.sparse import load_npz
from kmerexpr import transcriptome_reader as tr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def test_k1():
    ISO_FILE = os.path.join(DATA_PATH, "test1.fsa")
    X_FILE = os.path.join(DATA_PATH, "x1_csr.npz")
    K1 = 1
    tr.transcriptome_to_x(K1, ISO_FILE, X_FILE)
    x = load_npz(X_FILE)
    os.remove(X_FILE)
    M, T = x.get_shape()
    assert M == 4**1
    assert T == 2
    assert x.count_nonzero() == 8
    assert x[0, 0] == 0.25
    assert x[1, 0] == 0.25
    assert x[2, 0] == 0.25
    assert x[3, 0] == 0.25
    assert x[0, 1] == pytest.approx(0.40, 1e-6)
    assert x[1, 1] == pytest.approx(0.20, 1e-6)
    assert x[2, 1] == pytest.approx(0.20, 1e-6)
    assert x[3, 1] == pytest.approx(0.20, 1e-6)


def test_k2():
    ISO_FILE = os.path.join(DATA_PATH, "test2.fsa")
    X_FILE = os.path.join(DATA_PATH, "x2_csr.npz")
    K2 = 2
    tr.transcriptome_to_x(K2, ISO_FILE, X_FILE)
    x = load_npz(X_FILE)
    os.remove(X_FILE)
    M, T = x.get_shape()
    assert M == 4**2
    assert T == 3
    assert x.count_nonzero() == 7
    assert x[0, 0] == pytest.approx(1.0 / 3.0, 1e-6)
    assert x[1, 0] == pytest.approx(1.0 / 3.0, 1e-6)
    assert x[5, 0] == pytest.approx(1.0 / 3.0, 1e-6)
    assert x[6, 1] == pytest.approx(1.0 / 3.0, 1e-6)
    assert x[10, 1] == pytest.approx(1.0 / 3.0, 1e-6)
    assert x[15, 2] == 1.0


def test_simplex():
    ISO_FILE = os.path.join(DATA_PATH, "test2.fsa")
    X_FILE = os.path.join(DATA_PATH, "x2_csr.npz")
    K2 = 2
    tr.transcriptome_to_x(K2, ISO_FILE, X_FILE)
    x = load_npz(X_FILE)
    os.remove(X_FILE)
    M, T = x.get_shape()
    rng = np.random.default_rng()
    theta = np.abs(rng.standard_normal(T, dtype=np.float32))
    theta /= np.sum(theta)  # theta now simplex
    y_hat = x.dot(theta)
    assert sum(y_hat) == pytest.approx(1.0, 1e-5)
