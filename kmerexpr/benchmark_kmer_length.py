import multinomial_model as mm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt
from utils import plot_general
from utils import run_model
from utils import get_path_names
from utils import save_results, load_results
import random



random.seed(42) # Fixing the seed to reproduce results
filename = "GRCh38_latest_rna.fna" # "test5"
Ks = [ 11, 13, 15]  # 1, 2,  3,  5, 7,
N = 200000  # Number of reads
L = 100  # length of read
## Generate data, with ground truth theta
READS_FILE = sr.simulate_reads(filename, N, L)  # Only do this once
ISO_FILE, READS_FILE, X_FILE, THETA_FILE = get_path_names(filename, N, L)

errors_lbfgs = []
# Ks, errors_lbfgs = load_results(filename, N, L)
n_repeat = 1

error_repeat = []
for i in range(n_repeat):
    for K in Ks:
        errors = run_model(ISO_FILE, READS_FILE, X_FILE, THETA_FILE, mm.multinomial_model, N, L, K)
        error_repeat.append(errors)
    errors_lbfgs.append(error_repeat)
    save_results(filename, N, L, Ks, errors_lbfgs)
    error_repeat = []
dict_plot = {}
dict_plot["softmax+lbfgs"] = errors_lbfgs
# saving and plotting results
save_results(filename, N, L,  Ks, errors_lbfgs)
plot_general(dict_plot, title=filename + "-N-" + str(N) + "-L-" + str(L) + "-Kmax-"+str(np.maximum(Ks)) , save_path="./figures", 
            yaxislabel=r"$\|\theta -\theta^* \|/\|\theta_0 -\theta^* \|$", xaxislabel="Kmer Length", xticks = Ks, logplot = False, miny =0)


