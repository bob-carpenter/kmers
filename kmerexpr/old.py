from transcriptome_reader import transcriptom_to_x

from scipy.sparse import csr_matrix, save_npz  # BSD-3
import numpy as np  # BSD-3
import fastaparser  # GPLv3

from collections import Counter
from functools import lru_cache


float_t = np.float32
int_t = np.int32

fasta_file = "data/GRCh38_latest_rna.fna"
print("fasta transcriptome file =", fasta_file)

K = 5
print("K =", K)

M = 4**K
print("M =", M)

max_nz = 400 * 1000 * 1000
print("maximum non-zeros =", max_nz)


def shred(seq, K):
    N = len(seq)
    for n in range(0, N - K + 1):
        yield seq[n : n + K]


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


with open(fasta_file) as fasta_file:
    parser = fastaparser.Reader(fasta_file, parse_method="quick")
    n = 0
    data = np.zeros(max_nz, dtype=float_t)
    row_ind = np.zeros(max_nz, dtype=int_t)
    col_ind = np.zeros(max_nz, dtype=int_t)
    pos = 0
    for seq in parser:
        if n % 1000 == 0:
            print("n = ", n)
        if "PREDICTED" in seq.header:
            continue
        iter = shred(seq.sequence, K)
        iter_filtered = filter(valid_kmer, iter)

        kmer_counts = Counter(iter_filtered)
        total_count = sum(kmer_counts.values())
        for (kmer, count) in kmer_counts.items():
            data[pos] = float_t(count / total_count)
            row_ind[pos] = int_t(kmer_to_id(kmer))
            col_ind[pos] = int_t(n)
            pos += 1
        n += 1

print("resizing")
data.resize(pos)
row_ind.resize(pos)
col_ind.resize(pos)

print("building csr_matrix")
xt = csr_matrix((data, (row_ind, col_ind)), shape=(M, n), dtype=float_t)

print("saving .npz")
save_npz(f"data/xt_{K}.npz", xt)


# TESTING AFTER HERE
rng = np.random.default_rng()
theta = np.abs(rng.standard_normal(n, dtype=np.float32))
theta /= np.sum(theta)  # theta now a simplex

y_hat = xt.dot(theta)
# print(y_hat)

print("sum(xt * theta) =", sum(y_hat), " [should be 1.0]")
