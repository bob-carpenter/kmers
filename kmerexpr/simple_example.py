import multinomial_model as mm
import multinomial_simplex_model as msm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import get_path_names
from utils import load_theta_true_and_theta_sampled
from plotting import plot_error_vs_iterations, plot_scatter_theta
import random
import time


if __name__ == '__main__': 
    random.seed(42) 

    # filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    # K = 1
    # N = 5000000
    # L = 70
    filename = "test5.fsa" # "test5.fsa" "GRCh38_latest_rna.fna"
    K = 2
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

    model = msm.multinomial_simplex_model(X_FILE, Y_FILE, beta = 0.1) # initialize model
    # Initialize theta0
    alpha = np.ones(model.T())
    theta0 = alpha/alpha.sum()

    tic = time.perf_counter()
    theta, f_sol, dict_results= model.fit(theta0, n_iters =100)
    toc = time.perf_counter()
    print(f"Fitting model took {toc - tic:0.4f} seconds")

    os.remove(X_FILE)  # delete X file
    os.remove(Y_FILE)  # delete Y file

    ## Plotting
    theta_true, theta_sampled = load_theta_true_and_theta_sampled(filename, N, L) # load the theta_true
    model_type = "simplex"
    title=filename + '-theta-errors-''-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K)
    plot_error_vs_iterations(dict_results, theta_true, title, model_type)

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
    theta_scat = dict_results['x']
    title = filename+'-'+model_type + "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K) 
    plot_scatter_theta(title,theta_scat,theta_true)
    plot_scatter_theta(title,theta_scat,theta_scat- theta_true, horizontal=True)