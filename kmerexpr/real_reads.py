import numpy as np  # BSD-3
import fastaparser  # GPLv3
from simulate_reads import length_adjustment, length_adjustment_inverse
from utils import save_simulation_parameters, save_lengths, get_simulation_dir
from os import path
from Bio import SeqIO
import re, collections
import numpy, pandas

import os

np.random.seed(42)


def polyester_parser(fasta, duds=""):
    Ns = []
    ncRNAs = []

    for record in SeqIO.parse(fasta, "fasta"):
        if "N" in record.seq:
            Ns.append(record.id)
        if ", mRNA" not in record.description:
            ncRNAs.append(record.id)

    seq_parse = SeqIO.to_dict(SeqIO.parse(fasta, "fasta"))

    remove = [key for key, value in seq_parse.items() if len(value) < 100]

    for small in remove:
        try:
            del seq_parse[small]
        except KeyError:
            pass

    for N in Ns:
        try:
            del seq_parse[N]
        except KeyError:
            pass

    for ncRNA in ncRNAs:
        try:
            del seq_parse[ncRNA]
        except KeyError:
            pass

    if duds:
        duds = open(duds).readlines()

        duds = [dud.replace(".fa", "") for dud in duds]

        for dud in duds:
            try:
                del seq_parse[dud]
            except KeyError:
                pass

    outputdir = os.path.dirname(fasta)

    with open(os.path.join(outputdir, "cleaned_fasta.fa"), "w") as handle:
        SeqIO.write(seq_parse.values(), handle, "fasta")


def simulate_reads(problem, force_repeat=True):  #
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
    if (
        path.exists(get_simulation_dir(problem))
        and path.exists(READS_FILE)
        and force_repeat is False
    ):
        return READS_FILE
    # if path.exists(get_simulation_dir(problem)) and force_repeat is False:
    #     print("Simulation results for ", problem ," already exists. To re-compute, pass the argument force_repeat = true in simulate_reads" )
    # elif path.exists(get_simulation_dir(problem)) is True:
    #     if path.exists(READS_FILE) and force_repeat is False: # don't repeat if not needed
    #         print("file ", READS_FILE, " already exists. To re-compute, pass the argument force_repeat = true in simulate_reads" )
    #         return READS_FILE
    T = 0
    isoforms_header = []
    lengths_list = []
    L = problem.L
    N = problem.N

    seq_list = [record.id for record in SeqIO.parse("sample_01.fasta", "fasta")]

    read_name = re.compile(r"\w*_\d*\.\d*")
    seq_str = " ".join(seq_list)
    tx_names = read_name.findall(seq_str)

    cnt = collections.Counter()
    for word in tx_names:
        cnt[word] += 1

    seq_dict = dict(cnt)

    N = len(tx_names)

    theta_dict = {}

    for seq_name, bins in seq_dict.items():
        theta_dict[seq_name] = bins / N

    seq_frame = pandas.DataFrame.from_dict(seq_dict, orient="index")
    theta_frame = pandas.DataFrame.from_dict(theta_dict, orient="index")

    combined = pandas.concat([seq_frame, theta_frame], axis=1)
    combined.columns = ["seq_len", "theta"]
    combined.to_csv("processed_reads.csv")

    lengths = tuple(combined["seq_len"])
    theta_sampled = tuple(combined["theta"])

    psi = length_adjustment(theta_sampled, lengths)

    # theta_sampled = bins/N
    # lengths = np.asarray(lengths_list)
    # psi = length_adjustment(theta_sampled, lengths)

    print("length list: ", lengths_list)

    print("adj ", lengths)
    print("psi ", psi)
    # theta_true= length_adjustment(psi, lengths)
    # print("theta true: ",theta_true)
    # print("theta[0:10] =", theta_true[0:10])
    # print("theta[K-10:K] =", theta_true[T - 10 : T])

    # iso_sampled = np.random.choice(T, size=N, replace=True, p=theta_true)
    # bins, bin_edges = np.histogram(y_sampled, bins=np.arange(T+1) )
    # bins = np.bincount(iso_sampled, minlength=T)
    # theta_sampled = bins/N
    # theta sampled
    save_simulation_parameters(problem, psi, theta_sampled, theta_sampled)
    save_lengths(problem.filename, N, L, lengths)
    with open(READS_FILE, "w") as out:
        for n in range(N):
            if (n + 1) % 100000 == 0:
                print("sim n = ", n + 1)
            # seq = isoforms[iso_sampled[n]]
            start = np.random.choice(len(seq) - L + 1)
            out.write(">sim-")
            # out.write(str(n)+"/"+isoforms_header[iso_sampled[n]][1:])
            out.write("\n")
            out.write(seq[start : start + L])
            out.write("\n")
    return READS_FILE


if __name__ == "__main__":
    from utils import Problem

    problem = Problem(filename="test5.fsa", K=8, N=1000, L=14)
    alpha = 0.1  # The parameter of the Dirchlet that generates readsforce_repeat = True
    ISO_FILE, READS_FILE, X_FILE, Y_FILE = problem.get_path_names()
    READS_FILE = simulate_reads(problem)
    print("generated: ", READS_FILE)
