import numpy as np  # BSD-3
import fastaparser  # GPLv3
from utils import get_path_prefix_surfix
from utils import save_simulation_parameters
from os import path


def sample_genome_data(filename, sampled_file, p):  #
    """
    filename:  name of data file
    p = percentage of sampled data to be extracted
    """
    ISO_FILE, prefix, surfix = get_path_prefix_surfix(filename, N=1, L=1) # dont worry about this K=1. It only affects the Y_FILE, which is not used here
    SAMPLE_FILE_PATH = ISO_FILE.replace(filename, sampled_file)
    isoforms = []
    with open(ISO_FILE, "r") as f:
        parser = fastaparser.Reader(f, parse_method="quick")
        pos = 0
        for s in parser:
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            if pos % 100000 == 0:
                print("sim seqs read = ", pos)
            isoforms.append(seq)
            pos += 1
    T = len(isoforms)
    print("isoforms found = ", T)
    sample_size = int(np.ceil(p*T))
    y_sampled = np.random.choice(T, size=sample_size, replace=False)
    # bins, bin_edges = np.histogram(y_sampled, bins=np.arange(T+1) )
    with open(SAMPLE_FILE_PATH, "w") as out:
        for n in range(sample_size): #range(T): Rob: Used to be number of isoforms, but that's incorrect? We are writing the reads file, which has N rows
            seq = isoforms[y_sampled[n]]
            out.write(">sim-")
            out.write(str(n))
            out.write("\n")
            out.write(seq)
            out.write("\n")
            # print("sampled isoform num ", y_sampled[n], " is ",  seq)
    return SAMPLE_FILE_PATH

if __name__ == '__main__': 
    p = 0.01
    filename = "../data/GRCh38_latest_rna.fna"
    sampled_file = "sampled_genome" + "_" + str(p)
    sample_genome_data(filename, sampled_file, p)



