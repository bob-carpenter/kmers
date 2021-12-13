# import numpy as np
# from scipy.sparse import load_npz

import transcriptome_reader as tr

K = 2
TRANSCRIPTOME_FILE='kmerexpr/test_data/test1.fsa'
X_FILE = f"kmerexpr/test_data/x_{K}_csr.npz"

def test_answer():
    tr.transcriptome_to_x(K, TRANSCRIPTOME_FILE, X_FILE)
    assert 1 == 1
    
# print("loading x from file")
# start_time = time.time()
# x = load_npz(X_FILE)
# M, N = x.get_shape()
# print("M =", M, ";  N =", N)
# end_time = time.time()
# print("x loaded in time =", end_time - start_time, "seconds")
# rng = np.random.default_rng()
# theta = np.abs(rng.standard_normal(N, dtype=np.float32))
# theta /= np.sum(theta)  # theta now an
# y_hat = x.dot(theta)
# start_time = time.time()
# print("sum(x * theta) =", sum(y_hat), " [should be 1.0]")
# end_time = time.time()
# print("multiply time =", end_time - start_time, "seconds")







