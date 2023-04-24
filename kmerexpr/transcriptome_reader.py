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


def _get_libkmers_handle():
    from ctypes import CDLL, POINTER, c_float, c_int, c_char_p, byref
    from ctypes.util import find_library

    libkmers_path = find_library('kmers')
    if not libkmers_path:
        import sys, os
        if sys.base_prefix != sys.prefix:  # in venv, use that as root
            libroot = sys.prefix
        else:  # take path relative to this script and hope for the best
            libroot = os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-5])

        if sys.platform == 'linux':
            extension = ".so"
        elif sys.platform == 'darwin':
            extension = ".dylib"
        elif sys.platform == 'win32':
            extension = '.dll'
        else:
            raise ImportError("Invalid operating platform found in 'libkmers' search")
        lib = os.path.join(libroot, "lib", f"libkmers{extension}")
        lib64 = os.path.join(libroot, "lib64", f"libkmers{extension}")
        if os.path.exists(lib):
            libkmers_path = lib
        elif os.path.exists(lib64):
            libkmers_path = lib64

    if not libkmers_path:
        raise ImportError("Unable to find 'libkmers' shared object")

    return CDLL(libkmers_path)


def transcriptome_to_x_fast(
        K,
        fasta_file,
        x_file,
        L=None,
        max_nz=500 * 1000 * 1000,
        float_t=np.float32,
        int_t=np.int32,
):
    if L is not None:
        raise ValueError("Only L=None currently supported for fast reader")
    if float_t != np.float32:
        raise ValueError("Only float_t=np.float32 currently supported for fast reader")
    if int_t != np.int32:
        raise ValueError("Only int_t=np.int32 currently supported for fast reader")
    print("K =", K)
    print("fasta file =", fasta_file)
    print("target x file =", x_file)
    print("float type =", float_t)
    print("int type =", int_t)
    M = 4**K
    print("M =", M)

    libkmers = _get_libkmers_handle()
    func = libkmers.fasta_to_kmers_sparse
    from ctypes import POINTER, c_float, c_int, c_char_p, byref
    func.restype = c_int
    func.argtypes = [
        c_char_p,
        c_int,
        POINTER(c_float),
        POINTER(c_int),
        POINTER(c_int),
        c_int,
        POINTER(c_int),
        POINTER(c_int),
    ]

    data = np.zeros(max_nz, dtype=float_t)
    row_ind = np.zeros(max_nz, dtype=int_t)
    col_ind = np.zeros(max_nz, dtype=int_t)
    pos = c_int(0)
    n_cols = c_int(0)

    failed = func(fasta_file.encode('UTF-8'),
                  K,
                  data.ctypes.data_as(POINTER(c_float)),
                  row_ind.ctypes.data_as(POINTER(c_int)),
                  col_ind.ctypes.data_as(POINTER(c_int)),
                  max_nz,
                  byref(pos),
                  byref(n_cols),
                  )
    pos = pos.value
    n_cols = n_cols.value

    if failed:
        raise RuntimeError("max_nz insufficient to load fasta file")

    print("trimming triplets")
    data = data[0:pos]
    row_ind = row_ind[0:pos]
    col_ind = col_ind[0:pos]
    print("building csr_matrix")
    xt = csr_matrix((data, (row_ind, col_ind)), shape=(M, n_cols), dtype=float_t)
    print("saving csr matrix to file = ", x_file)
    save_npz(x_file, xt)
