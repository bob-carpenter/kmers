import multinomial_model as mm
import multinomial_simplex_model as msm
import transcriptome_reader as tr
import simulate_reads as sr
from simulate_reads import length_adjustment_inverse
import numpy as np
from plotting import plot_general
from run_single import run_model, run_model_load_and_save
from utils import save_run_result, load_simulation_parameters
from utils import save_kmer_length_results, load_kmer_length_results, load_run_result
import random
import matplotlib.pyplot as plt

random.seed(42) # Fixing the seed to reproduce results
model_type = "simplex" 
# model_type = "softmax"

if(model_type == "softmax"):
    model = mm.multinomial_model
else:
    model = msm.multinomial_simplex_model

filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
Ks = [1,  3,  7, 9, 11]
N = 1000
L = 14
# filename =  "GRCh38_latest_rna.fna"
# Ks = [ 1, 2,  3,  5, 7, 11, 13, 14, 15]  #,14, 15
# N = 5000000  # Number of reads
# L = 80  # length of read

force_repeat = True
load_old = False
## Generate data, with ground truth theta
READS_FILE = sr.simulate_reads(filename, N, L, force_repeat=force_repeat)  # Only do this once
dict_simulation = load_simulation_parameters(filename, N, L)
theta_true = dict_simulation['theta_true']
psi_true = dict_simulation['psi']
# Ks_o, errors_lbfgs_o = load_results(filename, N, L)
n_repeat = 1

for i in range(n_repeat):
    for K in Ks:
        print("K: ", K)
        # theta_opt = run_model_load_and_save(filename, model, N, L, K, load_old = load_old, n_iters= 200, force_repeat = force_repeat)
        dict_opt = run_model(filename, model, N, L, K, n_iters= 200)
        save_run_result(filename, N, L,  K, dict_opt)


# plot results
error_true_list =[]
error_true = []
for K in Ks:
    dict_results = load_run_result(filename, N, L,  K)
    theta_opt = dict_results['x']
    psi_opt = length_adjustment_inverse(theta_opt, dict_simulation['lengths'])
    err = np.linalg.norm(psi_true - psi_opt, ord=1) # / np.linalg.norm(theta_true - theta0)
    error_true.append(err)
error_true_list.append(error_true)
dict_plot = {}
# errors, Ks = load_kmer_length_results(filename, N, L)
dict_plot[model_type ] = error_true_list
# saving and plotting error vs Ks
plot_general(dict_plot, title=filename +'-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-Kmax-"+str(np.max(Ks)) , save_path="./figures", 
            yaxislabel=r"$\|\psi^{opt} -\psi^{*} \|$", xaxislabel="Kmer Length", xticks = Ks, logplot = False, miny =0)
plt.close()
