import multinomial_model as mm
import multinomial_simplex_model as msm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
from plotting import plot_general
from run_single import run_model, run_model_load_and_save
from utils import save_run_result, load_theta_true_and_theta_sampled
from utils import save_kmer_length_results, load_kmer_length_results
import random
import matplotlib.pyplot as plt

random.seed(42) # Fixing the seed to reproduce results
model_type = "simplex" 
# model_type = "softmax"

if(model_type == "softmax"):
    model = mm.multinomial_model
else:
    model = msm.multinomial_simplex_model

# filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
# Ks = [1,  3,  7, 9, 11]
# N = 1000
# L = 14
filename =  "GRCh38_latest_rna.fna"
Ks = [ 1, 2,  3,  5, 7, 11, 13, 14, 15]  #,14, 15
N = 5000000  # Number of reads
L = 100  # length of read


## Generate data, with ground truth theta
READS_FILE = sr.simulate_reads(filename, N, L)  # Only do this once
theta_true, theta_sampled = load_theta_true_and_theta_sampled(filename, N, L)
# Ks_o, errors_lbfgs_o = load_results(filename, N, L)
n_repeat = 1

error_true = []
error_true_list = []
for i in range(n_repeat):
    for K in Ks:
        print("K: ", K)
        theta_opt = run_model_load_and_save(filename, model, N, L, K, load_old = True, n_iters= 1000)
        # theta_opt,  f_sol, dict_sol = run_model(filename, model, N, L, K, n_iters= 1000)
        # save_run_result(filename, N, L,  K, dict_sol)
        err = np.linalg.norm(theta_true - theta_opt, ord=1) # / np.linalg.norm(theta_true - theta0)
        error_true.append(err)
        save_kmer_length_results(filename+'-'+model_type, N, L, Ks, error_true_list)
        
    error_true_list.append(error_true)
    error_true = []

dict_plot = {}
errors, Ks = load_kmer_length_results(filename, N, L)
dict_plot[model_type ] = error_true_list
# saving and plotting error vs Ks
save_kmer_length_results(filename+'-'+model_type, N, L, Ks, error_true_list)
plot_general(dict_plot, title=filename +'-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-Kmax-"+str(np.max(Ks)) , save_path="./figures", 
            yaxislabel=r"$\|\theta^{opt} -\theta^{*} \|$", xaxislabel="Kmer Length", xticks = Ks, logplot = False, miny =0)
plt.close()
