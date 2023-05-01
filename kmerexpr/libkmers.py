import sys
import os
import numpy as np
from ctypes import CDLL, POINTER, c_float, c_int, c_uint64, c_char_p, byref
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
_common_args = [
    c_int,
    POINTER(c_char_p),
    c_int,
    c_int,
    POINTER(c_float),
    POINTER(c_uint64),
    POINTER(c_uint64),
    POINTER(c_int),
    c_uint64,
    POINTER(c_uint64),
    POINTER(c_int),
]

_fasta_to_kmers_sparse = _libkmers.fasta_to_kmers_sparse
_fasta_to_kmers_sparse.restype = c_int
_fasta_to_kmers_sparse.argtypes = _common_args

_fasta_to_kmers_csr = _libkmers.fasta_to_kmers_csr
_fasta_to_kmers_csr.restype = c_int
_fasta_to_kmers_csr.argtypes = _common_args

_fasta_to_kmers_sparse_cat_subseq = _libkmers.fasta_to_kmers_sparse_cat_subseq
_fasta_to_kmers_sparse_cat_subseq.restype = c_int
_fasta_to_kmers_sparse_cat_subseq.argtypes = _common_args

_fasta_to_kmers_csr_cat_subseq = _libkmers.fasta_to_kmers_csr_cat_subseq
_fasta_to_kmers_csr_cat_subseq.restype = c_int
_fasta_to_kmers_csr_cat_subseq.argtypes = _common_args

_fastq_to_kmers_sparse = _libkmers.fastq_to_kmers_sparse
_fastq_to_kmers_sparse.restype = c_int
_fastq_to_kmers_sparse.argtypes = _common_args


def _get_total_size(files):
    tot_size = 0
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError("Invalid FASTA path provided")
        tot_size += os.stat(file).st_size
    return tot_size


def fasta_to_kmers_sparse(fasta_files, K, max_nz=0, L=0, concatenate_subseq=False):
    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]

    tot_size = _get_total_size(fasta_files)
    if max_nz == 0:
        max_nz = tot_size

    M = 4**K
    data = np.zeros(max_nz, dtype=np.float32)
    row_ind = np.zeros(max_nz, dtype=np.uint64)
    col_ind = np.zeros(max_nz, dtype=np.uint64)
    kmer_counts = np.zeros(M, dtype=np.int32)

    pos = c_uint64(0)
    n_cols = c_int(0)

    files = (c_char_p * (len(fasta_files)))()
    files[:] = [file.encode('UTF-8') for file in fasta_files]
    if concatenate_subseq:
        failed = _fasta_to_kmers_sparse_cat_subseq(len(fasta_files),
                                                   files,
                                                   K,
                                                   L,
                                                   data.ctypes.data_as(POINTER(c_float)),
                                                   row_ind.ctypes.data_as(POINTER(c_uint64)),
                                                   col_ind.ctypes.data_as(POINTER(c_uint64)),
                                                   kmer_counts.ctypes.data_as(POINTER(c_int)),
                                                   max_nz,
                                                   byref(pos),
                                                   byref(n_cols),
                                                   )
    else:
        failed = _fasta_to_kmers_sparse(len(fasta_files),
                                        files,
                                        K,
                                        L,
                                        data.ctypes.data_as(POINTER(c_float)),
                                        row_ind.ctypes.data_as(POINTER(c_uint64)),
                                        col_ind.ctypes.data_as(POINTER(c_uint64)),
                                        kmer_counts.ctypes.data_as(POINTER(c_int)),
                                        max_nz,
                                        byref(pos),
                                        byref(n_cols),
                                        )

    if failed == -1:
        raise RuntimeError("Failed to open FASTA file")
    if failed == -2:
        raise RuntimeError("Failed to parse FASTA file: insufficient max_nz")

    pos = pos.value
    n_cols = n_cols.value

    data = data[0:pos]
    row_ind = row_ind[0:pos]
    col_ind = col_ind[0:pos]

    return data, row_ind, col_ind, kmer_counts, n_cols


def fasta_to_kmers_csr(fasta_files, K, max_nz=0, L=0, concatenate_subseq=False):
    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]

    tot_size = _get_total_size(fasta_files)
    if max_nz == 0:
        max_nz = tot_size

    M = 4**K
    data = np.zeros(max_nz, dtype=np.float32)
    col_ind = np.zeros(max_nz, dtype=np.uint64)
    row_ind = np.zeros(M + 1, dtype=np.uint64)
    kmer_counts = np.zeros(M, dtype=np.int32)

    pos = c_uint64(0)
    n_cols = c_int(0)

    files = (c_char_p * (len(fasta_files)))()
    files[:] = [file.encode('UTF-8') for file in fasta_files]
    if concatenate_subseq:
        failed = _fasta_to_kmers_csr_cat_subseq(len(fasta_files),
                                                files,
                                                K,
                                                L,
                                                data.ctypes.data_as(POINTER(c_float)),
                                                row_ind.ctypes.data_as(POINTER(c_uint64)),
                                                col_ind.ctypes.data_as(POINTER(c_uint64)),
                                                kmer_counts.ctypes.data_as(POINTER(c_int)),
                                                max_nz,
                                                byref(pos),
                                                byref(n_cols),
                                                )
    else:
        failed = _fasta_to_kmers_csr(len(fasta_files),
                                     files,
                                     K,
                                     L,
                                     data.ctypes.data_as(POINTER(c_float)),
                                     row_ind.ctypes.data_as(POINTER(c_uint64)),
                                     col_ind.ctypes.data_as(POINTER(c_uint64)),
                                     kmer_counts.ctypes.data_as(POINTER(c_int)),
                                     max_nz,
                                     byref(pos),
                                     byref(n_cols),
                                     )

    if failed == -1:
        raise RuntimeError("Failed to open FASTA file")
    if failed == -2:
        raise RuntimeError("Failed to parse FASTA file: insufficient max_nz")

    pos = pos.value
    n_cols = n_cols.value

    data = data[0:pos]
    col_ind = col_ind[0:pos]

    return data, row_ind, col_ind, kmer_counts, n_cols



def fastq_to_kmers_sparse(fastq_files, K, max_nz=0, L=0):
    tot_size = _get_total_size(fastq_files)
    if max_nz == 0:
        max_nz = tot_size

    M = 4**K
    data = np.zeros(max_nz, dtype=np.float32)
    row_ind = np.zeros(max_nz, dtype=np.uint64)
    col_ind = np.zeros(max_nz, dtype=np.uint64)
    kmer_counts = np.zeros(M, dtype=np.int32)

    pos = c_uint64(0)
    n_cols = c_int(0)

    files = (c_char_p * (len(fastq_files)))()
    files[:] = [file.encode('UTF-8') for file in fastq_files]
    failed = _fastq_to_kmers_sparse(len(fastq_files),
                                    files,
                                    K,
                                    L,
                                    data.ctypes.data_as(POINTER(c_float)),
                                    row_ind.ctypes.data_as(POINTER(c_uint64)),
                                    col_ind.ctypes.data_as(POINTER(c_uint64)),
                                    kmer_counts.ctypes.data_as(POINTER(c_int)),
                                    max_nz,
                                    byref(pos),
                                    byref(n_cols),
                                    )
    if failed:
        raise RuntimeError("Failed to parse FASTQ file(s)")

    pos = pos.value
    n_cols = n_cols.value

    data = data[0:pos]
    row_ind = row_ind[0:pos]
    col_ind = col_ind[0:pos]

    return data, row_ind, col_ind, kmer_counts, n_cols
