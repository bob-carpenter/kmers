import multinomial_model as mm
import multinomial_simplex_model as msm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import get_path_names
from utils import load_theta_true_and_theta_sampled
from utils import save_run_result, load_run_result
from utils import merge_run_model_Dictionaries
import random
import time


def run_model_load_and_save(filename, model_class, N, L, K, load_old = False, n_iters = 5000, force_repeat = True):
    # Fixing the seed to reproduce results

    dict_old = None
    if load_old:
        dict_old = load_run_result(filename, N, L,  K)
    theta_opt,  f_sol_div_d0, dict_new = run_model(filename, model_class, N, L, K, n_iters = n_iters, dict_old = dict_old, force_repeat = force_repeat) # , batchsize= "full"
    theta_true, theta_sampled = load_theta_true_and_theta_sampled(filename, N, L)
    print('Distance theta_opt to theta_true: ', np.linalg.norm(theta_opt - theta_true, ord =1))
    print('Distance theta_av to theta_true: ', np.linalg.norm(dict_new['x_av'] - theta_true, ord =1))

    if load_old:
        dict_new = merge_run_model_Dictionaries(dict_old, dict_new)  # merge previous dictionary with new dictionary
    save_run_result(filename, N, L,  K, dict_new) # saving latest solution
    return theta_opt



def run_model(filename, model_class, N, L, K,  n_iters = 5000, batchsize= None, dict_old =None, force_repeat = True):
   # Need to check if y and X already exit. And if so, just load them.
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = get_path_names(filename, N, L, K)
    
    if dict_old == None:
        theta0 = None
        continue_from = 0
    else:
        theta0 = dict_old['x']
        continue_from = dict_old["iteration_counts"][-1]

    tic = time.perf_counter()
    if os.path.exists(Y_FILE) is False or force_repeat is True:
        y = reads_to_y(K, READS_FILE, Y_FILE=Y_FILE)
    if os.path.exists(X_FILE) is False or force_repeat is True:
        tr.transcriptome_to_x(K, ISO_FILE, X_FILE,  L  =L)
    toc = time.perf_counter()
    ## Fit model
    model = model_class(X_FILE, Y_FILE)

    print(f"Generated/loaded y and X  in {toc - tic:0.4f} seconds")
    if theta0 is None:
        if model.name == "mirror": # We should be using alpha parameter less than one, push towards boundaries
            alpha = np.ones(model.T())
            theta0 = alpha/alpha.sum()
        else:
            theta0 = np.random.normal(0, 1, model.T()) 

    tic = time.perf_counter()
    theta, f_sol, dict_new= model.fit(theta0, n_iters =n_iters, batchsize = None, continue_from = continue_from)
    toc = time.perf_counter()
    print(f"Fitting model took {toc - tic:0.4f} seconds")

    # os.remove(X_FILE)  # delete file
    return theta,  f_sol, dict_new


if __name__ == '__main__': 
    random.seed(42) 
    model_type = "simplex" 
    # model_type = "softmax"

    if(model_type == "softmax"):
        model_class = mm.multinomial_model
    else:
        model_class = msm.multinomial_simplex_model

    filename = "GRCh38_latest_rna.fna" # "test5.fsa" "GRCh38_latest_rna.fna"
    K = 1
    N = 5000000
    L = 70
    
    tic = time.perf_counter()
    READS_FILE = sr.simulate_reads(filename, N, L)  # force_repeat=True to force repeated simulation
    toc = time.perf_counter()
    print(f"Created reads in {toc - tic:0.4f} seconds")

    run_model_load_and_save(filename, model_class, N, L, K, load_old = True, n_iters = 1000)

