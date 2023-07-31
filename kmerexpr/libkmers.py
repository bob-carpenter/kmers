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
    c_uint64,
    POINTER(c_uint64),
    POINTER(c_int),
]

_fasta_count_kmers = _libkmers.fasta_count_kmers
_fasta_count_kmers.restype = c_int
_fasta_count_kmers.argtypes = [
    c_int,
    POINTER(c_char_p),
    c_int,
    POINTER(c_int),
]

_fastq_gz_count_kmers = _libkmers.fastq_gz_count_kmers
_fastq_gz_count_kmers.restype = c_int
_fastq_gz_count_kmers.argtypes = [
    c_int,
    POINTER(c_char_p),
    c_int,
    POINTER(c_int),
]

_fasta_to_kmers_csr = _libkmers.fasta_to_kmers_csr
_fasta_to_kmers_csr.restype = c_int
_fasta_to_kmers_csr.argtypes = _common_args

_fasta_to_kmers_csr_cat_subseq = _libkmers.fasta_to_kmers_csr_cat_subseq
_fasta_to_kmers_csr_cat_subseq.restype = c_int
_fasta_to_kmers_csr_cat_subseq.argtypes = _common_args


def _get_total_size(files):
    tot_size = 0
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError("Invalid FASTA path provided")
        tot_size += os.stat(file).st_size
    return tot_size


def fasta_count_kmers(fasta_files, K):
    if isinstance(fasta_files, str):
        fasta_files = [fasta_files]
    for file in fasta_files:
        if not os.path.isfile(file):
            raise FileNotFoundError("Invalid FASTA path provided")

    M = 4**K
    counts = np.zeros(M, dtype=np.int32)
    files = (c_char_p * (len(fasta_files)))()
    files[:] = [file.encode('UTF-8') for file in fasta_files]
    failed = _fasta_count_kmers(len(fasta_files),
                                files,
                                K,
                                counts.ctypes.data_as(POINTER(c_int)),
                                )
    if failed == -1:
        raise RuntimeError("Failed to open FASTA file")

    return counts


def fastq_gz_count_kmers(fastq_files, K):
    if isinstance(fastq_files, str):
        fastq_files = [fastq_files]
    for file in fastq_files:
        if not os.path.isfile(file):
            raise FileNotFoundError("Invalid fastq.gz path provided")

    M = 4**K
    counts = np.zeros(M, dtype=np.int32)
    files = (c_char_p * (len(fastq_files)))()
    files[:] = [file.encode('UTF-8') for file in fastq_files]
    failed = _fastq_gz_count_kmers(len(fastq_files),
                                   files,
                                   K,
                                   counts.ctypes.data_as(POINTER(c_int)),
                                   )
    if failed == -1:
        raise RuntimeError("Failed to open FASTQ file")

    return counts


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

    return data, row_ind, col_ind, n_cols
