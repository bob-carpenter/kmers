import time
st = time.time()
import os
import kmerexpr.transcriptome_reader as tr
from kmerexpr import simulate_reads as sr
from kmerexpr.simulate_reads import length_adjustment_inverse
from kmerexpr.libkmers import fasta_count_kmers
from kmerexpr.multinomial_simplex_model import multinomial_simplex_model as MSM
from kmerexpr.utils import (load_lengths, Problem, Model_Parameters,
                            load_simulation_parameters, get_plot_title)
from kmerexpr.plotting.plots import plot_error_vs_iterations, plot_scatter
import numpy as np

K = 15  # size of k-mers to use
L = 100
N = 50000000
ISO_FILE = os.path.expanduser('/mnt/home/rblackwell/ceph/kmers/GRCh38.fna')
X_FILE = os.path.expanduser(f'/mnt/home/rblackwell/ceph/kmers/GRCh38_K{K}.npz')
Y_FILE = os.path.expanduser(f'/mnt/home/rblackwell/ceph/kmers/GRCh38_K{K}_counts.npz')
READS_FILE = os.path.expanduser(f'/mnt/home/rgower/ceph/kmers/GRCh38_L{L}_N{N}_READS.npz')

x = tr.transcriptome_to_x(K, ISO_FILE, L = L)
mask = np.array(np.sum(x,axis=1)).squeeze() == 0
problem = Problem(filename="GRCh38.fna", K=K, N=N, L=L, alpha =0.1)
# Need to pass the ISO_FILE
READS_FILE, lengths = sr.simulate_reads(problem, READS_FILE, ISO_FILE) 
y = fasta_count_kmers(READS_FILE, problem.K)
# y = np.random.randint(0, 100, x.shape[0])
y[mask] = 0
params = Model_Parameters("")
# lengths = load_lengths(problem.filename, problem.N, problem.L)
model = params.initialize_model(x, y, lengths=lengths)  # initialize model. beta =1 is equivalent to no prior/regularization

dict_results = model.fit(params, n_iters=2000)

## Plotting and checking against ground truth
dict_simulation = load_simulation_parameters(problem)
theta_true = dict_simulation["theta_true"]
theta_sampled = dict_simulation["theta_sampled"]
psi_true = dict_simulation["psi"]

title = get_plot_title(problem, params)
if params.model_type == "simplex":
    title_errors = title + "-theta-errors-"
    plot_error_vs_iterations(
        dict_results, theta_true, title_errors, model_type="simplex"
    )

# Plotting scatter of theta_{opt} vs theta_{*} for a fixed k
theta_opt = dict_results["x"]
psi_opt = length_adjustment_inverse(theta_opt, lengths)
plot_scatter(title, psi_opt, psi_true)
plot_scatter(title, psi_opt, psi_opt - psi_true, horizontal=True)




