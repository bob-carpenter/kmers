from scipy.sparse import csr_matrix
import fastaparser   # GPLv3
from collections import Counter
from collections import deque
from itertools import islice
import numpy as np

K = 4
M = 4**K

def shred(seq, K):
    N = len(seq)
    for n in range(0, N - K + 1):
        yield seq[n:n + K]

BASE_ID_CODE = {'A':0, 'C':1, 'G': 2, 'T':3}

def base_to_code(b):
    return BASE_ID_CODE[b]
        
def kmer_to_id(kmer):
    id = 0;
    for c in kmer:
        id = 4 * id + base_to_code(c)
    return id

def valid_kmer(kmer):
    for c in kmer:
        if (c != 'A' and c != 'C' and c != 'G' and c != 'T'):
            return False
    return True

with open("data/GRCh38_latest_rna.fna") as fasta_file:
    parser = fastaparser.Reader(fasta_file, parse_method='quick')
    n = 0
    data = list()
    row_ind = list()
    col_ind = list()
    for seq in parser:
        if (n % 100 == 0):
            print("n = ", n)
        if "PREDICTED" in seq.header:
            continue
        iter = shred(seq.sequence, K)
        iter_filtered = filter(valid_kmer, iter)
        kmer_counts = Counter(iter_filtered)
        total_count = sum(kmer_counts.values())
        for (kmer, count) in kmer_counts.items():
            data.append(count / total_count)
            row_ind.append(kmer_to_id(kmer))
            col_ind.append(n)
        n += 1

print("building csr_matrix")
xt = csr_matrix((data, (row_ind, col_ind)), shape = (M, n))

rng = np.random.default_rng()
theta = np.abs(rng.standard_normal(n))
theta /= np.sum(theta)

y_hat = xt.dot(theta)
print(y_hat)
        

    
    


