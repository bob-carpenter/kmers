import numpy as np  # BSD-3
from utils import save_simulation_parameters, save_lengths, get_simulation_dir
from os import path
import fasta

def length_adjustment(psi, lengths):
    theta = psi*lengths
    theta = theta/theta.sum()
    return theta

def length_adjustment_inverse(theta, lengths):
    psi = theta/lengths
    psi = psi/psi.sum()
    return psi

def simulate_reads(problem,  force_repeat = True):  #
    np.random.seed(42)
    """
    Simulates reads from a given reference isoforms.  First subsamples the isoforms (represents biological sample),
    then samples the reads from this subsampled data.
    problem.filename: returns a string that will be the surfix of generated data where
                "read"+filename will be reads data
                "x"+filename the transciptome matrix
                "theta"+filename the ground truth theta used to generate the data
    problem.L = length of read
    problem.N = number of reads, must be greater than number of isoforms in ISO_FILE
    problem.alpha = parameter of Dirchlet distribution that generates psi
    """
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
    if path.exists(get_simulation_dir(problem)) and path.exists(READS_FILE) and force_repeat is False:
        return READS_FILE
    # if path.exists(get_simulation_dir(problem)) and force_repeat is False:
    #     print("Simulation results for ", problem ," already exists. To re-compute, pass the argument force_repeat = true in simulate_reads" )
    # elif path.exists(get_simulation_dir(problem)) is True:
    #     if path.exists(READS_FILE) and force_repeat is False: # don't repeat if not needed
    #         print("file ", READS_FILE, " already exists. To re-compute, pass the argument force_repeat = true in simulate_reads" )
    #         return READS_FILE
    isoforms = []
    isoforms_header = []
    lengths_list = []
    L = problem.L
    N = problem.N
    with open(ISO_FILE, "r") as f:
        parser = fasta.read_fasta(f)
        pos = 0
        for s in parser:
            print("s: ",s)
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            print("seq: ",seq)
            if len(seq) < L:
                continue
            if pos % 100000 == 0:
                print("sim seqs read = ", pos)
            isoforms.append(seq)
            isoforms_header.append(s.header)
            lengths_list.append(len(seq)-L+1)
            pos += 1
    T = len(isoforms)
    print("isoforms found = ", T)
    alphas = problem.alpha*np.ones(T)
    psi= np.random.dirichlet(alphas)
    print("length list: ",lengths_list)
    lengths = np.asarray(lengths_list)
    print("adj ",lengths)
    print("psi ",psi)
    theta_true= length_adjustment(psi, lengths)
    print("theta true: ",theta_true)
    print("theta[0:10] =", theta_true[0:10])
    print("theta[K-10:K] =", theta_true[T - 10 : T])
    iso_sampled = np.random.choice(T, size=N, replace=True, p=theta_true)
    # bins, bin_edges = np.histogram(y_sampled, bins=np.arange(T+1) )
    bins = np.bincount(iso_sampled, minlength=T)
    theta_sampled = bins/N
    save_simulation_parameters(problem, psi, theta_true, theta_sampled)
    save_lengths(problem.filename, N, L, lengths)
    with open(READS_FILE, "w") as out:
        for n in range(N):
            if (n + 1) % 100000 == 0:
                print("sim n = ", n + 1)
            seq = isoforms[iso_sampled[n]]
            start = np.random.choice(len(seq) - L + 1)
            out.write(">sim-")
            out.write(str(n)+"/"+isoforms_header[iso_sampled[n]][1:])
            out.write("\n")
            out.write(seq[start : start + L])
            out.write("\n")
    return READS_FILE


if __name__ == '__main__':

    from utils import Problem
    problem = Problem(filename="test5.fsa", K=8, N =1000, L=14)
    alpha = 0.1  #The parameter of the Dirchlet that generates readsforce_repeat = True
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
    READS_FILE = simulate_reads(problem)
    print("generated: ", READS_FILE)

