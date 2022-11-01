
import multinomial_model as mm
import multinomial_simplex_model as msm
import normal_model as mnm
import transcriptome_reader as tr
import simulate_reads as sr
from rna_seq_reader import reads_to_y
import numpy as np
import os 
from utils import save_run_result
import random
import time

from simulate_reads import length_adjustment_inverse
from utils import load_lengths, Problem, Model_Parameters
from utils import load_simulation_parameters, load_run_result, get_plot_title
from plotting import plot_error_vs_iterations, plot_scatter 

def plot_errors_and_scatter(problem, model_parameters):
    dict_results = load_run_result(problem, model_parameters)
    dict_simulation = load_simulation_parameters(problem)
    theta_true = dict_simulation['theta_true']
    psi_true = dict_simulation['psi']
    # Plotting error vs iterations for simplex method
    title = get_plot_title(problem, model_parameters)
    if model_parameters.model_type == "simplex":
        title_errors =title + '-theta-errors'
        plot_error_vs_iterations(dict_results, theta_true, title_errors, model_parameters.model_type)

    # Plotting scatter of psi_{opt} vs psi_{*}
    lengths = load_lengths(problem.filename, problem.N, problem.L)
    theta_opt = dict_results['x']
    psi_opt = length_adjustment_inverse(theta_opt, lengths)

    plot_scatter(title+"-theta", theta_opt, dict_simulation['theta_sampled'], horizontal=False)
    plot_scatter(title+"-theta", theta_opt,  theta_opt-dict_simulation['theta_sampled'], horizontal=True)
    
    plot_scatter(title, psi_opt, psi_true, horizontal=False)
    plot_scatter(title, psi_opt,  psi_opt-psi_true, horizontal=True)    
    RMSE = np.sqrt(np.linalg.norm(psi_true - psi_opt)/psi_true.shape)
    print("MSE distance to psi_true = ", str(RMSE))
    print("L1 distance to psi_true = ", str(np.linalg.norm(psi_true - psi_opt, ord =1)))
    print("L1 distance to theta_true = ", str(np.linalg.norm(theta_true - theta_opt, ord =1)))
    print("L1 distance to theta_sampled = ", str(np.linalg.norm(dict_simulation['theta_sampled'] - theta_opt, ord =1)))

def run_model(problem, model_parameters, n_iters = 5000,   force_repeat = False):
   # Need to check if y and X already exit. And if so, just load them.
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
    tic = time.perf_counter()
    if os.path.exists(Y_FILE) is False or force_repeat is True:
        print("Generating ", Y_FILE)
        y = reads_to_y(problem.K, READS_FILE, Y_FILE=Y_FILE)
    if os.path.exists(X_FILE) is False or force_repeat is True:
        print("Generating ", X_FILE)
        tr.transcriptome_to_x(problem.K, ISO_FILE, X_FILE,  L  =problem.L)
    toc = time.perf_counter()

    print(f"Generated/loaded y and X  in {toc - tic:0.4f} seconds")
    tic = time.perf_counter()
    ## Fit model
    lengths = load_lengths(problem.filename, problem.N, problem.L)
    print("Fittting model: ", model_parameters.model_type)
    model = model_parameters.initialize_model(X_FILE, Y_FILE,  lengths=lengths)
    dict_results = model.fit(model_parameters, n_iters=n_iters)
    toc = time.perf_counter()
    dict_results['fit-time'] = toc - tic
    print(f"Fitting model took {toc - tic:0.4f} seconds")

    # os.remove(X_FILE)  # delete file
    return dict_results


if __name__ == '__main__': 
    random.seed(42) 
    # model_parameters = Model_Parameters(model_type = "softmax", solver_name = "lbfgs",  lrs = "warmstart", init_iterates ="uniform")
    model_parameters = Model_Parameters(model_type = "simplex", solver_name = "exp_grad", lrs = "warmstart")
    # model_parameters = Model_Parameters(model_type = "simplex", solver_name = "frank_wolfe", lrs = "warmstart", init_iterates ="uniform")
    problem = Problem(filename="GRCh38_latest_rna.fna", K=15, N =50000000, L=100, alpha =0.1) #N =5000000, 10000000, 20000000, 50000000
    # problem = Problem(filename="sampled_genome_"+str(0.01), K=15, N =5000000, L=100,  alpha=0.1)  # alpha=0.1, 0.5
    # problem = Problem(filename="test5.fsa", K=8, N =1000, L=14, alpha=10)
    # problem = Problem(filename="GRCh38_latest_rna.fna", K=14, N=50000000, L=100, alpha = 0.1) 

    force_repeat = False
    print("experiment (N, L, K) = (",str(problem.N),", ",str(problem.L), ", ",str(problem.K), ")" )
    tic = time.perf_counter()
    sr.simulate_reads(problem, force_repeat=force_repeat)  # force_repeat=True to force repeated simulation

    toc = time.perf_counter()
    print(f"Created reads in {toc - tic:0.4f} seconds")
    try:
        dict_results = load_run_result(problem, model_parameters)
    except:
        dict_results= run_model(problem, model_parameters, n_iters = 400)
        save_run_result(problem, model_parameters, dict_results)
    # dict_results = run_model(problem, model_parameters, n_iters = 400, force_repeat = force_repeat) 
    save_run_result(problem, model_parameters, dict_results) # saving latest solution
    dict_results = load_run_result(problem, model_parameters)
    dict_simulation = load_simulation_parameters(problem)

    plot_errors_and_scatter(problem, model_parameters)


