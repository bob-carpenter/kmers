import numpy as np
from collections import Counter
from kmerexpr.transcriptome_reader import valid_kmer, shred, kmer_to_id
from kmerexpr import fasta
from scipy.sparse import load_npz, save_npz, coo_matrix


def reads_to_y(K, READS_FILE, float_t=np.float32, int_t=np.int32, Y_FILE=None):
    print("K =", K)
    print("fasta file =", READS_FILE)
    print("float type =", float_t)
    print("int type =", int_t)
    M = 4**K
    print("M =", M)
    y = np.zeros(M)
    n = 0
    with open(READS_FILE) as f:
        parser = fasta.read_fasta(f)
        for seq in parser:
            if n % 10000 == 0:
                print("seqs read =", n)
            iter = shred(seq.sequence, K)
            iter_filtered = filter(valid_kmer, iter)
            kmer_counts = Counter(iter_filtered)
            for kmer, count in kmer_counts.items():
                id = kmer_to_id(kmer)
                y[id] += count
            n += 1
    if Y_FILE is not None:
        save_npz(Y_FILE, coo_matrix(y))
    return y


def load_xy(x_file, y_file):
    x = load_npz(x_file)
    if isinstance(y_file, np.ndarray):
        y = y_file
    else:
        y = load_npz(y_file).toarray().squeeze()
    return x, y
