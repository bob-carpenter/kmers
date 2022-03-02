import numpy as np  # BSD-3
import fastaparser  # GPLv3
from utils import get_path_names


def simulate_reads(file_name, N, L):  #
    """
    file_name:  a string that will be the surfix of generated data where
                "read"+file_name will be reads data
                "x"+file_name the transciptome matrix
                "theta"+file_name the ground truth theta used to generate the data
    L = length of read
    N = number of reads, must be greater than number of isoforms in ISO_FILE
    """
    ISO_FILE, READS_FILE, X_FILE, THETA_FILE = get_path_names(file_name, N, L)
    isoforms = []
    with open(ISO_FILE, "r") as f:
        parser = fastaparser.Reader(f, parse_method="quick")
        pos = 0
        for s in parser:
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            if len(seq) < L:
                continue
            if pos % 10000 == 0:
                print("seqs read = ", pos)
            isoforms.append(seq)
            pos += 1
    T = len(isoforms)
    print("isoforms found = ", T)
    # if(T < N):
    #     print("WARNING!!!: number of reads ", N, " is greater than the number of isoforms ", T)
    beta = np.random.uniform(0, 1)
    alpha = beta* np.ones(T)
    theta = np.random.dirichlet(alpha)
    print("theta[0:10] =", theta[0:10])
    print("theta[K-10:K] =", theta[T - 10 : T])
    np.save(THETA_FILE, theta)
    y = np.random.choice(T, size=N, replace=True, p=theta)
    with open(READS_FILE, "w") as out:
        for n in range(N): #range(T): Rob: Used to be number of isoforms, but that's incorrect? We are writing the reads file, which has N rows
            if (n + 1) % 100000 == 0:
                print("sim n = ", n + 1)
            seq = isoforms[y[n]]
            start = np.random.choice(len(seq) - L + 1)
            out.write(">sim-")
            out.write(str(n))
            out.write("\n")
            out.write(seq[start : start + L])
            out.write("\n")
    return READS_FILE
