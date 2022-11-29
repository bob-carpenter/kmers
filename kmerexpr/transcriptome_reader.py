from scipy.sparse import csr_matrix, save_npz, load_npz  # BSD-3
import numpy as np  # BSD-3
from collections import Counter
from functools import lru_cache
from kmerexpr import fasta

base_id = {"A": 0, "C": 1, "G": 2, "T": 3}


def base_to_code(b):
    return base_id[b]


@lru_cache(maxsize=None)
def kmer_to_id(kmer):
    id = 0
    for c in kmer:
        id = 4 * id + base_to_code(c)
    return id


@lru_cache(maxsize=None)
def valid_kmer(kmer):
    for c in kmer:
        if c not in "ATCG":
            return False
    return True


def shred(seq, K):
    N = len(seq)
    for n in range(0, N - K + 1):
        yield seq[n : n + K]


def transcriptome_to_x(
    K,
    fasta_file,
    x_file,
    L=None,
    max_nz=500 * 1000 * 1000,
    float_t=np.float32,
    int_t=np.int32,
):
    print("K =", K)
    print("fasta file =", fasta_file)
    print("target x file =", x_file)
    print("float type =", float_t)
    print("int type =", int_t)
    M = 4 ** K
    print("M =", M)
    with open(fasta_file) as f:
        parser = fasta.read_fasta(f)
        n = 0
        data = np.zeros(max_nz, dtype=float_t)
        row_ind = np.zeros(max_nz, dtype=int_t)
        col_ind = np.zeros(max_nz, dtype=int_t)
        pos = 0
        for s in parser:
            if n % 10000 == 0:
                print("seqs iso = ", n)
            if "PREDICTED" in s.header:
                continue
            seq = s.sequence
            if L is not None:
                if len(seq) < L:
                    continue
            iter = shred(s.sequence, K)
            iter_filtered = filter(valid_kmer, iter)
            kmer_counts = Counter(iter_filtered)
            total_count = sum(kmer_counts.values())
            for (kmer, count) in kmer_counts.items():
                data[pos] = float_t(count / total_count)
                row_ind[pos] = int_t(kmer_to_id(kmer))
                col_ind[pos] = int_t(n)
                pos += 1
            n += 1
    print("trimming triplets")
    data.resize(pos)
    row_ind.resize(pos)
    col_ind.resize(pos)
    print("building csr_matrix")
    xt = csr_matrix((data, (row_ind, col_ind)), shape=(M, n), dtype=float_t)
    print("saving csr matrix to file = ", x_file)
    save_npz(x_file, xt)
