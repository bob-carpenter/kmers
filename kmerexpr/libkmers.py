import sys
import os
import numpy as np
from ctypes import CDLL, POINTER, c_float, c_int, c_char_p, byref
from ctypes.util import find_library

def _get_libkmers_handle():
    libkmers_path = find_library('kmers')
    if not libkmers_path:
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


_libkmers = _get_libkmers_handle()
_fasta_to_kmers_sparse = _libkmers.fasta_to_kmers_sparse
_fasta_to_kmers_sparse.restype = c_int
_fasta_to_kmers_sparse.argtypes = [
    c_char_p,
    c_int,
    POINTER(c_float),
    POINTER(c_int),
    POINTER(c_int),
    POINTER(c_int),
    c_int,
    POINTER(c_int),
    POINTER(c_int),
]

def fasta_to_kmers_sparse(fasta_file, K, max_nz):
    M = 4**K
    data = np.zeros(max_nz, dtype=np.float32)
    row_ind = np.zeros(max_nz, dtype=np.int32)
    col_ind = np.zeros(max_nz, dtype=np.int32)
    kmer_counts = np.zeros(M, dtype=np.int32)

    pos = c_int(0)
    n_cols = c_int(0)

    failed = _fasta_to_kmers_sparse(fasta_file.encode('UTF-8'),
                                    K,
                                    data.ctypes.data_as(POINTER(c_float)),
                                    row_ind.ctypes.data_as(POINTER(c_int)),
                                    col_ind.ctypes.data_as(POINTER(c_int)),
                                    kmer_counts.ctypes.data_as(POINTER(c_int)),
                                    max_nz,
                                    byref(pos),
                                    byref(n_cols),
                                    )
    if failed:
        raise RuntimeError("Failed to parse FASTA file")

    pos = pos.value
    n_cols = n_cols.value

    data = data[0:pos]
    row_ind = row_ind[0:pos]
    col_ind = col_ind[0:pos]

    return data, row_ind, col_ind, kmer_counts, n_cols
