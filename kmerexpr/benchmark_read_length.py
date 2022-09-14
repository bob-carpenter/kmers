
from rna_seq_reader import reads_to_y
import simulate_reads as sr
import numpy as np
from scipy.special import softmax as softmax
from matplotlib import pyplot as plt
from utils import  Problem, Model_Parameters, save_run_result
from utils import load_simulation_parameters
from plotting import plot_general
import random
from run_single import run_model
from utils import get_errors

random.seed(42) # Fixing the seed
filename = str(5)
Ns = [1, 50, 100]
Ls = [1, 3, 5, 7, 10]
Ks = [1, 2, 3, 4]
N = 200  # Number of reads
L = 10  # length of read
model_type = "simplex"
solver_name = "exp_grad"
lrs = None #"lin-warmstart"
filename= "test5.fsa"  # "GRCh38_latest_rna.fna"   "sampled_genome_"+str(0.1)
## Generate data, with ground truth theta
K = 3  # K-mer length
n_repeat = 1
errors_total = []
error_repeat = []
for i in range(n_repeat):
    for N in Ns:
        model_parameters = Model_Parameters(model_type =model_type, solver_name=solver_name, lrs=lrs )
        problem = Problem(filename=filename, K=K, N =N, L=L)
        READS_FILE = sr.simulate_reads(problem, force_repeat=False) 
        dict_results = run_model(problem, model_parameters, n_iters = 50)
        dict_simulation = load_simulation_parameters(problem)
        error= np.linalg.norm(dict_results['x'] - dict_simulation['theta_true'], ord =1) 
        error_repeat.append(error)
        save_run_result(problem, model_parameters, dict_results) 
    errors_total.append(error_repeat)
    error_repeat = []
# os.remove(X_FILE)   ## Keeping the file for reproducibility
dict_plot = {}
dict_plot[model_type] = errors_total
# import pdb; pdb.set_trace()
plot_general(dict_plot, title="test"+filename + "-L-" + str(L) + "-reads", save_path="./figures", yaxislabel=r"$\|\theta -\theta^* \|$", xaxislabel="Num. Reads", xticks = Ns)

