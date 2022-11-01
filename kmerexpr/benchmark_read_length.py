
from rna_seq_reader import reads_to_y
import simulate_reads as sr
from simulate_reads import length_adjustment_inverse
import numpy as np
from scipy.special import softmax as softmax
from matplotlib import pyplot as plt
from utils import  Problem, Model_Parameters, save_run_result
from utils import load_simulation_parameters, load_lengths
from plotting import plot_general, plot_general_test
import random
from run_single import run_model, plot_errors_and_scatter
from utils import get_errors, load_run_result
import os

random.seed(42) # Fixing the seed
# Ns = [1, 100, 200, 500, 1000]
# Ls = [8, 12, 14]
# Ks = [ 3, 6, 8]
Ns = [1000000, 5000000, 10000000, 50000000]
Ls = [100, 200, 300]
Ks = [8, 12, 15]
alphas = [0.01, 0.1, 1., 10.0]
model_type = "simplex"
solver_name = "exp_grad"
lrs = "armijo"
# filename= "test5.fsa"  # "GRCh38_latest_rna.fna"   "sampled_genome_"+str(0.1)  
# filename= "sampled_genome_"+str(0.1)
filename= "GRCh38_latest_rna.fna"
## Generate data, with ground truth theta
n_repeat = 1
fig, axs = plt.subplots(len(Ls), len(alphas), figsize=(11, 11))
# for i in range(n_repeat):
for Lcount, L in enumerate(Ls):
    for alphacount, alpha in enumerate(alphas):
        dict_plot = {}
        error_reads = []
        for K in Ks:
            errors_total = []
            for N in Ns:
                model_parameters = Model_Parameters(model_type =model_type, solver_name=solver_name, lrs=lrs )
                problem = Problem(filename=filename, K=K, N =N, L=L, alpha =alpha)

                sr.simulate_reads(problem, force_repeat=False) 
                dict_simulation = load_simulation_parameters(problem)
                try:
                    dict_results = load_run_result(problem, model_parameters)
                except:
                    dict_results= run_model(problem, model_parameters, n_iters = 200)
                    save_run_result(problem, model_parameters, dict_results)

                # error= np.linalg.norm(dict_results['x'] - dict_simulation['theta_true'], ord =1) 
                lengths = load_lengths(problem.filename, problem.N, problem.L)
                psi_opt = length_adjustment_inverse(dict_results['x'], lengths)
                RMSE = np.sqrt(np.linalg.norm(dict_simulation['psi'] - psi_opt)/np.sqrt(psi_opt.shape))
                error_reads.append(RMSE)
                save_run_result(problem, model_parameters, dict_results) 
                plot_errors_and_scatter(problem, model_parameters)
            errors_total.append(error_reads)
            error_reads = []
            # os.remove(X_FILE)   ## Keeping the file for reproducibility
            dict_plot["K ="+str(K)] = errors_total
        plt.sca(axs[Lcount,alphacount])
        if Lcount == 0:
            axs[Lcount,alphacount].set_title(r"$\alpha$ =" + str(alpha), fontsize=20)
        # axs[Lcount,alphacount].set_title("L = " + str(L) +  r", $\alpha$ =" + str(alpha), fontsize=20)
        plot_general(dict_plot, title="benchmark_read_length_"+filename + "-L-" + str(L) + "-reads-"+ str(alpha) +"-alpha", \
             save_path="./figures", yaxislabel="L = " + str(L), xaxislabel="N. Reads", xticks = Ns)  #r"$\|\psi -\psi^* \|/T$"
        plt.close()

yminabs = 100000
ymaxabs = 0
# Adjusting legends, labels and y-axis limits
for ax in axs.flat:
    ymin, ymax = ax.get_ylim()
    print("ymin: ", ymin," ymax: ", ymax)
    yminabs = np.minimum(ymin,yminabs)
    ymaxabs = np.maximum(ymax,ymaxabs)
    ax.label_outer()
    legend = ax.legend()
    legend.remove()
plt.setp(ax,  ylim=(yminabs, ymaxabs))

handles, labels = axs[len(Ls)-1,len(alphas)-1].get_legend_handles_labels()
axs[len(Ls)-1,len(alphas)-1].legend(handles, labels, fontsize=20) 
fig.savefig("./figures/benchmark_read_length_"+filename+".pdf", bbox_inches="tight", pad_inches=0.01)