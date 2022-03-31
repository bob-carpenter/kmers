import multinomial_model as mm
import multinomial_simplex_model as msm
import normal_model as mnm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import get_path_names
from utils import save_run_result
import random
import time

from simulate_reads import length_adjustment_inverse
from plotting import plot_scatter, plot_error_vs_iterations
from utils import load_simulation_parameters
from utils import load_run_result

def plot_errors_and_scatter(filename, model_type, N, L, K):
    dict_results = load_run_result(filename, model_type, N, L,  K)
    dict_simulation = load_simulation_parameters(filename, N, L)
    
    psi_true = dict_simulation['psi']
    # Plotting error vs iterations for simplex method
    if model_type == "simplex":
        theta_true = dict_simulation['theta_true']
        title=filename + '-theta-errors-''-'+model_type+ "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K)
        plot_error_vs_iterations(dict_results, theta_true, title, model_type)

    # Plotting scatter of theta_{opt} vs theta_{*} for a fixed k    
    theta_opt = dict_results['x']
    psi_opt = length_adjustment_inverse(theta_opt, dict_simulation['lengths'])
    title = filename+'-'+model_type + "-N-" + str(N) + "-L-" + str(L) + "-K-"+str(K) 
    plot_scatter(title,psi_opt,psi_true, horizontal=False)
    plot_scatter(title,psi_opt,psi_opt- psi_true, horizontal=True)
    MSE = np.linalg.norm(psi_true - psi_opt)/psi_true.shape
    print("MSE distance to solution = ", str(MSE))
    print("L1 distance to solution = ", str(np.linalg.norm(psi_true - psi_opt, ord =1)))


def run_model(filename, model_class, N, L, K,  n_iters = 5000, batchsize= None,  force_repeat = True):
   # Need to check if y and X already exit. And if so, just load them.
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = get_path_names(filename, N, L, K)
    tic = time.perf_counter()
    if os.path.exists(Y_FILE) is False or force_repeat is True:
        print("Generating ", Y_FILE)
        y = reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
    if os.path.exists(X_FILE) is False or force_repeat is True:
        print("Generating ", X_FILE)
        tr.transcriptome_to_x(K, ISO_FILE, X_FILE,  L  =L)
    toc = time.perf_counter()
    ## Fit model
    model = model_class(X_FILE, Y_FILE)
    print(f"Generated/loaded y and X  in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    if model_class == "simplex": # 
        alpha = np.ones(model.T())
        theta0 = alpha/alpha.sum()
        dict_opt= model.fit(theta0=None, n_iters =n_iters, batchsize = None)
    else:
        dict_opt = model.fit(theta0=None, n_iters =n_iters)
    
    
    toc = time.perf_counter()
    print(f"Fitting model took {toc - tic:0.4f} seconds")

    # os.remove(X_FILE)  # delete file
    return dict_opt


if __name__ == '__main__': 
    random.seed(42) 
    # model_type = "simplex" 
    model_type = "softmax"
    # model_type = "normal"

    if(model_type == "softmax"):
        model_class = mm.multinomial_model
    elif(model_type == "normal"):
        model_class = mnm.normal_model
    else:
        model_class = msm.multinomial_simplex_model


    filename =  "test5.fsa"# "test5.fsa" "GRCh38_latest_rna.fna"
    K = 6
    N = 2000
    L = 14
    # filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    # # K = 8
    # # N = 5000000
    # # L = 100  # waiting for this one 2nd tab with softmax, 
    # K = 10
    # N = 5000000
    # L = 70  # waiting for this one 3rd tab with softmax?
    force_repeat = False
    tic = time.perf_counter()
    READS_FILE = sr.simulate_reads(filename, N, L, force_repeat = force_repeat)  # force_repeat=True to force repeated simulation
    toc = time.perf_counter()
    print(f"Created reads in {toc - tic:0.4f} seconds")

    dict_opt = run_model(filename, model_class, N, L, K, n_iters = 200, force_repeat = force_repeat) # , batchsize= "full"
    save_run_result(filename, model_type, N, L,  K, dict_opt) # saving latest solution

    plot_errors_and_scatter(filename, model_type, N, L, K)


