import os

from kmerexpr import transcriptome_reader as tr
from kmerexpr import simulate_reads as sr
from kmerexpr.libkmers import fasta_count_kmers
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.utils import (load_lengths, Problem, Model_Parameters,
                            load_simulation_parameters, get_plot_title)
from kmerexpr.plotting.plots import plot_error_vs_iterations, plot_scatter
import random
import time

random.seed(42)

model_parameters = Model_Parameters(model_type="simplex")
problem = Problem(filename="test5.fsa", K=8, N=5000, L=14)

alpha = 0.1  # The parameter of the Dirchlet that generates readsforce_repeat = True
ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
tic = time.perf_counter()
READS_FILE = sr.simulate_reads(problem, READS_FILE, ISO_FILE)   # force_repeat=True to force repeated simulation

# Create y and X and save to file
x = tr.transcriptome_to_x(problem.K, ISO_FILE, L=problem.L)
toc = time.perf_counter()
print(f"Created reads, counts and transciptome matrix x in {toc - tic:0.4f} seconds")
y = fasta_count_kmers(READS_FILE, problem.K)

lengths = load_lengths(problem.filename, problem.N, problem.L)
model = model_parameters.initialize_model(x, y, lengths=lengths)  # initialize model. beta =1 is equivalent to no prior/regularization

tic = time.perf_counter()
dict_results = model.fit(model_parameters, n_iters=50)
toc = time.perf_counter()
print(f"Fitting model took {toc - tic:0.4f} seconds")

## Plotting and checking against ground truth
dict_simulation = load_simulation_parameters(problem)
theta_true = dict_simulation["theta_true"]
theta_sampled = dict_simulation["theta_sampled"]
psi_true = dict_simulation["psi"]

title = get_plot_title(problem, model_parameters)
if model_parameters.model_type == "simplex":
    title_errors = title + "-theta-errors-"
    plot_error_vs_iterations(
        dict_results, theta_true, title_errors, model_type="simplex"
    )

# Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
theta_opt = dict_results["x"]
psi_opt = length_adjustment_inverse(theta_opt, lengths)
plot_scatter(title, psi_opt, psi_true)
plot_scatter(title, psi_opt, psi_opt - psi_true, horizontal=True)
