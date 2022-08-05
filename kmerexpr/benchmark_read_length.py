import multinomial_model as mm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os
from scipy import optimize
from scipy.special import softmax as softmax
from scipy.sparse import load_npz
from matplotlib import pyplot as plt
from utils import plot_general
from utils import run_model
from utils import get_path_names
import random

random.seed(42) # Fixing the seed


filename = str(5)
Ns = [1, 10, 20, 50, 100, 200]
Ls = [1, 3, 5, 7, 10]
Ks = [1, 2, 3, 4]
N = 200  # Number of reads
L = 10  # length of read
## Generate data, with ground truth theta
# sr.simulate_reads(filename, N, L)  # Only do this once
ISO_FILE, X_FILE, THETA_FILE = get_path_names(filename, N, L)

K = 3  # K-mer length
n_repeat = 10
errors_lbfgs = []
error_repeat = []
for i in range(n_repeat):
    for N in Ns:
        errors = run_model(ISO_FILE, X_FILE, THETA_FILE, mm.multinomial_model, N, L, K)
        error_repeat.append(errors)
    errors_lbfgs.append(error_repeat)
    error_repeat = []
# os.remove(X_FILE)   ## Keeping the file for reproducibility
dict_plot = {}
dict_plot["lbfgs"] = errors_lbfgs
plot_general(dict_plot, title="test"+filename + "-L-" + str(L) + "-reads", save_path="./figures", yaxislabel=r"$\|\theta -\theta^* \|$", xaxislabel="Num. Reads", xticks = Ns)

