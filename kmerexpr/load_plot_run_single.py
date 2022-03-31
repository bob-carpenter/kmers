import multinomial_model as mm
import multinomial_simplex_model as msm
import numpy as np
from simulate_reads import length_adjustment_inverse
from plotting import plot_scatter_theta, plot_error_vs_iterations
from utils import load_simulation_parameters
from utils import load_run_result
import random
import time



def plot_errors_and_scatter(filename, model_type, N, L, K):

    dict_results = load_run_result(filename, model_type, N, L,  K)
    dict_simulation = load_simulation_parameters(filename, N, L)
    theta_true = dict_simulation['theta_true']
    psi_true = dict_simulation['psi']
    # Plotting error vs iterations for simplex method

    if model_type == "simplex":
        title=filename + '-theta-errors-''-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K)
        plot_error_vs_iterations(dict_results, theta_true, title, model_type)

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
    if(model_type == "softmax"):
        theta_opt = dict_results['x']
        psi_opt = length_adjustment_inverse(theta_opt, dict_simulation['lengths'])
    title = filename+'-'+model_type + "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K) 
    plot_scatter_theta(title,psi_opt,psi_true)
    plot_scatter_theta(title,psi_opt,psi_opt- psi_true, horizontal=True)

if __name__ == '__main__': 

    random.seed(42) 
    # model_type = "simplex" 
    model_type = "softmax"

    # filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    # K = 15  #,14, 15
    # N = 5000000  # Number of reads
    # L = 80 
    filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
    K = 4
    N = 1000
    L = 14

    plot_errors_and_scatter(filename, model_type, N, L)

    

