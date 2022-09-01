
import multinomial_model as mm
import multinomial_simplex_model as msm
import normal_model as mnm
import transcriptome_reader as tr
import simulate_reads as sr
from simulate_reads import length_adjustment_inverse
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import get_path_names, load_lengths
from utils import load_simulation_parameters, load_run_result
from plotting import plot_error_vs_iterations, plot_scatter, get_plot_title
import random
import time
import scipy
random.seed(42) 

model_type = "simplex" 
# model_type = "softmax"
# model_type = "normal"

if(model_type == "softmax"):
    model_class = mm.multinomial_model
elif(model_type == "normal"):
    model_class = mnm.normal_model
else:
    model_class = msm.multinomial_simplex_model
# solver = "experimental_lbfgs"
# solver = "mirror_lbfgs"
solver = "exp_grad"
# solver = "accel_mirror"
# solver = "frank_wolfe"
# filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
# K = 15
# N = 5000000
# L = 100

# p=0.1
# filename ="sampled_genome_"+str(p)
# K = 15
# N = 5000000
# L = 100 

filename = "test5.fsa" # "test5.fsa" "GRCh38_latest_rna.fna"
K = 11
N = 1000
L = 14

alpha = 0.1
force_repeat = True
ISO_FILE, READS_FILE, X_FILE, Y_FILE = get_path_names(filename, N, L, K, alpha=alpha)
tic = time.perf_counter()
READS_FILE = sr.simulate_reads(filename, N, L, alpha = alpha, force_repeat=force_repeat)  # force_repeat=True to force repeated simulation
dict_simulation = load_simulation_parameters(filename, N, L, alpha= alpha)
# Create y and X and save to file 
reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
tr.transcriptome_to_x(K, ISO_FILE, X_FILE,  L  =L)
toc = time.perf_counter()
print(f"Created reads, counts and transciptome matrix x in {toc - tic:0.4f} seconds")

lengths = load_lengths(filename, N, L)
beta = 0.1
# beta = 1.0  # Which means no regularization
model = model_class(X_FILE, Y_FILE, beta = beta, lengths=lengths, solver=solver) # initialize model. beta =1 is equivalent to no prior/regularization

tic = time.perf_counter()
dict_results= model.fit(n_iters =100, tol=1e-16, gtol=1e-16,  lrs = "linesearch", Hessinv = False) #lrs= "linesearch"
toc = time.perf_counter()
print(f"Fitting model took {toc - tic:0.4f} seconds")

## Plotting
theta_true  = dict_simulation['theta_true']
theta_sampled   = dict_simulation['theta_sampled']
psi_true = dict_simulation['psi']


title = get_plot_title(filename,model_type,N,L,K, solver)

if model_type=='simplex':
    title_errors=title +'-theta-errors-'
    plot_error_vs_iterations(dict_results, theta_true, title_errors, model_type)

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