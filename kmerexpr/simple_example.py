import multinomial_model as mm
import multinomial_simplex_model as msm
import transcriptome_reader as tr
import simulate_reads as sr
from simulate_reads import length_adjustment_inverse
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import get_path_names
from utils import load_simulation_parameters
from plotting import plot_error_vs_iterations, plot_scatter_theta
import random
import time


if __name__ == '__main__': 
    random.seed(42) 

    # filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    # K = 10
    # N = 5000000
    # L = 70
    filename = "test5.fsa" # "test5.fsa" "GRCh38_latest_rna.fna"
    K = 3
    N = 1000
    L = 14

    tic = time.perf_counter()
    READS_FILE = sr.simulate_reads(filename, N, L, force_repeat=True)  # force_repeat=True to force repeated simulation
    toc = time.perf_counter()
    print(f"Created reads in {toc - tic:0.4f} seconds")

    ISO_FILE, READS_FILE, X_FILE, Y_FILE = get_path_names(filename, N, L, K)
    # Create y and X and save to file 
    reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
    tr.transcriptome_to_x(K, ISO_FILE, X_FILE,  L  =L)

    model = msm.multinomial_simplex_model(X_FILE, Y_FILE, beta = 0.01) # initialize model
    # Initialize theta0
    alpha = np.ones(model.T())
    theta0 = alpha/alpha.sum()

    tic = time.perf_counter()
    theta, f_sol, dict_results= model.fit(theta0, n_iters =2000)
    toc = time.perf_counter()
    print(f"Fitting model took {toc - tic:0.4f} seconds")

    os.remove(X_FILE)  # delete X file
    os.remove(Y_FILE)  # delete Y file

    ## Plotting
    dict_simulation = load_simulation_parameters(filename, N, L)
    theta_true  = dict_simulation['theta_true']
    psi_true = dict_simulation['psi']

 
    # theta_true, theta_sampled = load_theta_true_and_theta_sampled(filename, N, L) # load the theta_true
    model_type = "simplex"
    title = filename+'-'+model_type + "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K) 
    title_errors=title +'-theta-errors-'
    plot_error_vs_iterations(dict_results, theta_true, title_errors, model_type)

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
    theta_opt = dict_results['x']
    psi_opt = length_adjustment_inverse(theta_opt, dict_simulation['lengths'])
    plot_scatter_theta(title,psi_opt,psi_true)
    plot_scatter_theta(title,psi_opt,psi_opt- psi_true, horizontal=True)