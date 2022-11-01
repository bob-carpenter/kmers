import simulate_reads as sr
import numpy as np
from scipy.special import softmax as softmax
from matplotlib import pyplot as plt
from utils import  Problem, Model_Parameters, save_run_result
from utils import load_simulation_parameters
from plotting import plot_general
import random
from run_single import run_model, plot_errors_and_scatter
from utils import get_errors, load_run_result


def solver_name_map(solver_name):
    abbreviated_name = solver_name.replace("exp_grad", "EG")
    abbreviated_name = abbreviated_name.replace("frank_wolfe", "FW")
    if "EG" in abbreviated_name:
        abbreviated_name = abbreviated_name.replace("warmstart", "KL")
    return abbreviated_name

random.seed(42) # Fixing the seed
model_type = "simplex"
solver_names = ["exp_grad", "frank_wolfe"]
# solver = "frank_wolfe"
# solver = "lbfgs"

lrs_list = [None, "warmstart", "armijo"]
alpha =0.1 # 0.1, , 10
problem = Problem(filename="GRCh38_latest_rna.fna", K=12, N =50000000, L=200, alpha = alpha) #N =5000000, 10000000, 20000000, 50000000
# problem = Problem(filename="sampled_genome_"+str(0.1), K=14, N =5000000, L=100,  alpha=alpha) 
# problem = Problem(filename="test5.fsa", K=8, N =1000, L=14, alpha=alpha)

## Get ground truth theta
try:
    dict_simulation = load_simulation_parameters(problem)
except:
    READS_FILE = sr.simulate_reads(problem)  # force_repeat=True to force repeated simulation
    dict_simulation = load_simulation_parameters(problem)

dict_plot = {}
n_repeat = 1
n_iters = 400
max_length= 0

for solver_name in solver_names:
    for lrs in lrs_list:
        print(solver_name+'-'+str(lrs))
        if lrs=="armijo" and solver_name == "frank_wolfe":
            continue
        error_repeat = []
        for i in range(n_repeat):
            model_parameters = Model_Parameters(model_type = model_type, solver_name = solver_name, lrs = lrs, init_iterates ="uniform")    
            try:
                dict_results = load_run_result(problem, model_parameters)
            except:
                dict_results= run_model(problem, model_parameters, n_iters = n_iters)
                save_run_result(problem, model_parameters, dict_results)
            
            errors= get_errors(dict_results['xs'], dict_simulation['theta_true']) #np.linalg.norm(dict_results['xs'] - dict_simulation['theta_true'], ord =1) 
            error_repeat.append(errors)
            max_length = np.maximum(max_length, len(errors))
        plot_name = solver_name 
        if lrs !=None:
            plot_name = plot_name +'-'+str(lrs)        
        plot_name = solver_name_map(plot_name)
        dict_plot[plot_name]= error_repeat
        plot_errors_and_scatter(problem, model_parameters)
            # dict_plot[solver_name_map(solver_name)] = error_repeat
        # else:
            # dict_plot[solver_name_map(solver_name)+'-'+str(lrs)] = error_repeat
 
iter_skip = np.maximum(int(n_iters/15),1)
xticks = iter_skip*np.arange(max_length)
title = "compare-"+problem.filename+ "-N-" + str(problem.N) \
            + "-L-" + str(problem.L) + "-K-"+str(problem.K) \
            +"-init-" +model_parameters.init_iterates +"-alpha-" +str(alpha)
plt.figure(figsize=(8, 8), dpi=80)
plot_general(dict_plot, title=title, save_path="./figures", yaxislabel=r"$\|\theta -\theta^* \|$", xaxislabel="iterations", xticks =xticks)
