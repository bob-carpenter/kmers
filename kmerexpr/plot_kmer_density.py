from kmerexpr import simulate_reads as sr
from kmerexpr.rna_seq_reader import reads_to_y
import numpy as np
from kmerexpr.utils import Problem
import random
import matplotlib.pyplot as plt
import os
random.seed(42) # Fixing the seed to reproduce results
alphas = [0.001, 0.01, 0.1, 1, 10]

# filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
# Ks = [1,  3,  7, 9, 11]
# N = 1000
# L = 14
filename =  "GRCh38_latest_rna.fna"
Ks = [1, 2,  3,  5, 7, 11, 13, 14, 15]  #,14, 15
N = 5000000  # Number of reads
L = 100  # length of read

force_repeat = True
load_old = False
for alpha in alphas:
    ynnzs = []
    ynnzs_relative = []
    for K in Ks:
        print("K: ", K)
        # theta_opt = run_model_load_and_save(filename, model, N, L, K, load_old = load_old, n_iters= 200, force_repeat = force_repeat)
        problem = Problem(filename=filename, K=K, N =N, L=L)
        ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
        READS_FILE = sr.simulate_reads(filename, N, L, alpha = alpha, force_repeat=force_repeat)  # Only do this once
        y = reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
        ynnzs.append(np.count_nonzero(y))
        ynnzs_relative.append(np.count_nonzero(y)/len(y))

    title = filename + "-N-" + str(N) + "-L-" + str(L) +'-alpha-'+str(alpha)
    plt.rc("text", usetex=True)
    plt.rc("font", family="sans-serif")
    plt.plot(Ks,ynnzs, markersize=12, lw=3)
    plt.title(title, fontsize=25)
    plt.xlabel("kmer length", fontsize=25)
    plt.ylabel("nnz(y)", fontsize=25)

    save_path="./figures"
    plt.savefig( os.path.join(save_path,"ynnz-"+ title + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close()

    plt.plot(Ks,ynnzs_relative, markersize=12, lw=3)
    plt.title(title, fontsize=25)
    plt.xlabel("kmer length", fontsize=25)
    plt.ylabel("nnz(y)/len(y)", fontsize=25)

    plt.savefig( os.path.join(save_path,"ynnz-ratio-"+ title + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    plt.close()