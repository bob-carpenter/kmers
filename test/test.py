"""testing, 1, 2, 3"""
import time

import numpy as np
from scipy.sparse import load_npz

from transcriptreader import transcriptome_to_x

K = 5
TRANSCRIPTOME_FILE = "data/GRCh38_latest_rna.fna"
X_FILE = f"data/xt_{K}.npz"

transcriptome_to_x(K, TRANSCRIPTOME_FILE, X_FILE)


print("loading x from file")
start_time = time.time()
x = load_npz(X_FILE)
M, N = x.get_shape()
print("M =", M, ";  N =", N)
end_time = time.time()
print("x loaded in time =", end_time - start_time, "seconds")
rng = np.random.default_rng()
theta = np.abs(rng.standard_normal(N, dtype=np.float32))
theta /= np.sum(theta)  # theta now an
y_hat = x.dot(theta)
start_time = time.time()
print("sum(x * theta) =", sum(y_hat), " [should be 1.0]")
end_time = time.time()
print("multiply time =", end_time - start_time, "seconds")


T: number of targets
M: number of kmers
x: M x T
theta: T x 1
