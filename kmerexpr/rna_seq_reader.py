import numpy as np
from collections import Counter
from transcriptome_reader import valid_kmer, shred, kmer_to_id
import fastaparser

def reads_to_y(K, fasta_file, float_t = np.float32, int_t = np.int32):
    print("K =", K)
    print("fasta file =", fasta_file)
    print("float type =", float_t)
    print("int type =", int_t)
    M = 4**K
    print("M =", M)
    y = np.zeros(M)
    n = 0
    with open(fasta_file) as f:
        parser = fastaparser.Reader(f, parse_method='quick')
        for seq in parser:
            if n % 10000 == 0:
                print("seqs read =", n)
            iter = shred(seq.sequence, K)
            iter_filtered = filter(valid_kmer, iter)
            kmer_counts = Counter(iter_filtered)
            for (kmer, count) in kmer_counts.items():
                id = kmer_to_id(kmer)
                y[id] += count
            n += 1
    return y
