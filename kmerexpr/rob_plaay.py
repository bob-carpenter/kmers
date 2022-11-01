
import multinomial_model as mm
import multinomial_simplex_model as msm
import normal_model as mnm
import transcriptome_reader as tr
import simulate_reads as sr
from simulate_reads import length_adjustment_inverse
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import load_lengths, Problem, Model_Parameters
from utils import load_simulation_parameters, load_run_result, get_plot_title
from plotting import plot_error_vs_iterations, plot_scatter 
import random
import time
import scipy
random.seed(42) 
# lrs = "armijo"
lrs = "warmstart"
# lrs = None
# beta = 0.1
# beta = 1.0  # Which means no regularization
# solver = "exp_grad"
solver = "frank_wolfe"
# solver = "lbfgs"
model_parameters = Model_Parameters(model_type = "simplex", solver_name = solver, lrs = lrs, init_iterates ="uniform") 
# model_parameters = Model_Parameters(model_type = "softmax")
# model_parameters = Model_Parameters(model_type = "normal")

# problem = Problem(filename="GRCh38_latest_rna.fna", K=15, N =10000000, L=100)
# problem = Problem(filename="sampled_genome_"+str(0.1), K=15, N =5000000, L=100, alpha=0.1)  #p=0.1
problem = Problem(filename="test5.fsa", K=8, N =1000, L=14, alpha=0.1)
force_repeat = True
ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
tic = time.perf_counter()
READS_FILE = sr.simulate_reads(problem, force_repeat=force_repeat)  # force_repeat=True to force repeated simulation
dict_simulation = load_simulation_parameters(problem)
# Create y and X and save to file 
if force_repeat:
    reads_to_y(problem.K, READS_FILE, Y_FILE=Y_FILE)
    tr.transcriptome_to_x(problem.K, ISO_FILE, X_FILE,  L  =problem.L)
toc = time.perf_counter()
print(f"Created reads, counts and transciptome matrix x in {toc - tic:0.4f} seconds")

lengths = load_lengths(problem.filename, problem.N, problem.L)
model = model_parameters.initialize_model(X_FILE, Y_FILE,  lengths=lengths) # initialize model. beta =1 is equivalent to no prior/regularization

tic = time.perf_counter()
dict_results= model.fit(model_parameters, n_iters =400) 
toc = time.perf_counter()
print(f"Fitting model took {toc - tic:0.4f} seconds")

## Plotting
theta_true  = dict_simulation['theta_true']
theta_sampled   = dict_simulation['theta_sampled']
psi_true = dict_simulation['psi']


title = get_plot_title(problem, model_parameters)

if model_parameters.model_type=='simplex':
    title_errors=title +'-theta-errors'
    plot_error_vs_iterations(dict_results, theta_true, title_errors, model_type = "simplex")

# Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
theta_opt = dict_results['x']
psi_opt = length_adjustment_inverse(theta_opt, lengths)
plot_scatter(title,psi_opt,psi_true)
plot_scatter(title,psi_opt,psi_opt- psi_true, horizontal=True)

# plot_scatter(title,theta_opt,theta_sampled)
# plot_scatter(title,theta_opt,theta_opt- theta_sampled, horizontal=True)

# Delete the data
# os.remove(X_FILE)  # delete X file
# os.remove(Y_FILE)  # delete Y file