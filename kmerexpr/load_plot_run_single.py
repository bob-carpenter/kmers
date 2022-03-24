import multinomial_model as mm
import multinomial_simplex_model as msm
import numpy as np

from plotting import plot_scatter_theta, plot_error_vs_iterations
from utils import load_theta_true_and_theta_sampled
from utils import load_run_result
import random
import time



if __name__ == '__main__': 
    
    model_type = "simplex" 
    # model_type = "softmax"

    if(model_type == "softmax"):
        model = mm.multinomial_model
    else:
        model = msm.multinomial_simplex_model

    filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    K = 15  #,14, 15
    N = 5000000  # Number of reads
    L = 80 
    # filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
    # K = 2
    # N = 1000
    # L = 14
    dict_results = load_run_result(filename, N, L,  K)
    theta_true, theta_sampled = load_theta_true_and_theta_sampled(filename, N, L)
    theta_comp = theta_true
    # Plotting error vs iterations for simplex method

    if model_type == "simplex":
        title=filename + '-theta-errors-''-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K)
        plot_error_vs_iterations(dict_results, theta_true, title, model_type)

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
    theta_scat = dict_results['x_av']
    title = filename+'-'+model_type + "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K) 
    plot_scatter_theta(title,theta_scat,theta_true)
    plot_scatter_theta(title,theta_scat,theta_scat- theta_true, horizontal=True)

