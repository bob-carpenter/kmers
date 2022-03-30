import numpy as np  # BSD-3
import fastaparser  # GPLv3
from utils import get_path_names
from utils import save_simulation_parameters
from os import path


def length_adjustment(psi, lengths):
    theta = psi*lengths
    theta = theta/theta.sum()
    return theta

def length_adjustment_inverse(theta, lengths):
    psi = theta/lengths
    psi = psi/psi.sum()
    return psi

def simulate_reads(filename, N, L, force_repeat=True, alpha = 1):  #
    """
    filename:  a string that will be the surfix of generated data where
                "read"+filename will be reads data
                "x"+filename the transciptome matrix
                "theta"+filename the ground truth theta used to generate the data
    L = length of read
    N = number of reads, must be greater than number of isoforms in ISO_FILE
    """
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = get_path_names(filename, N, L, K=1) # dont worry about this K=1. It only affects the Y_FILE, which is not used here
    if path.exists(READS_FILE) and force_repeat == False: # don't repeat if not needed
        print("file ", READS_FILE, " already exists. To re-compute, pass the argument force_repeat = true in simulate_reads" )
        return READS_FILE
    isoforms = []
    lengths_list = []
    with open(ISO_FILE, "r") as f:
        parser = fastaparser.Reader(f, parse_method="quick")
        pos = 0
        for s in parser:
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            if len(seq) < L:
                continue
            if pos % 100000 == 0:
                print("sim seqs read = ", pos)
            isoforms.append(seq)
            lengths_list.append(len(seq)-L+1)
            pos += 1
    T = len(isoforms)
    print("isoforms found = ", T)
    # beta = np.random.uniform(0, 1)
    alphas = alpha*np.ones(T)
    psi= np.random.dirichlet(alphas)
    lengths = np.asarray(lengths_list)
    theta_true= length_adjustment(psi, lengths)
    print("theta[0:10] =", theta_true[0:10])
    print("theta[K-10:K] =", theta_true[T - 10 : T])
    y_sampled = np.random.choice(T, size=N, replace=True, p=theta_true)
    # bins, bin_edges = np.histogram(y_sampled, bins=np.arange(T+1) )
    bins = np.bincount(y_sampled)
    # look for bincount function
    theta_sampled = bins/N
    save_simulation_parameters(filename, N, L, lengths, psi, theta_true, theta_sampled)
    with open(READS_FILE, "w") as out:
        for n in range(N): #range(T): Rob: Used to be number of isoforms, but that's incorrect? We are writing the reads file, which has N rows
            if (n + 1) % 100000 == 0:
                print("sim n = ", n + 1)
            seq = isoforms[y_sampled[n]]
            start = np.random.choice(len(seq) - L + 1)
            out.write(">sim-")
            out.write(str(n))
            out.write("\n")
            out.write(seq[start : start + L])
            out.write("\n")
    return READS_FILE
