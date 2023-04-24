from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz  # BSD-3
import numpy as np  # BSD-3
from ctypes.util import find_library
from ctypes import CDLL, POINTER, c_float, c_int, c_char_p, byref

def _get_libkmers_handle():
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


def load_xy(x_file, y_file):
    x = load_npz(x_file)
    if isinstance(y_file, np.ndarray):
        y = y_file
    else:
        y = load_npz(y_file).toarray().squeeze()
    return x, y


def transcriptome_to_x_y(K, fasta_file, x_file, y_file=None, L=None,
                         max_nz=500 * 1000 * 1000,
                         float_t=np.float32,
                         int_t=np.int32):
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
    func.restype = c_int
    func.argtypes = [
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

    data = np.zeros(max_nz, dtype=float_t)
    row_ind = np.zeros(max_nz, dtype=int_t)
    col_ind = np.zeros(max_nz, dtype=int_t)
    kmer_counts = np.zeros(M, dtype=int_t)

    pos = c_int(0)
    n_cols = c_int(0)

    failed = func(fasta_file.encode('UTF-8'),
                  K,
                  data.ctypes.data_as(POINTER(c_float)),
                  row_ind.ctypes.data_as(POINTER(c_int)),
                  col_ind.ctypes.data_as(POINTER(c_int)),
                  kmer_counts.ctypes.data_as(POINTER(c_int)),
                  max_nz,
                  byref(pos),
                  byref(n_cols),
                  )
    pos = pos.value
    n_cols = n_cols.value

    if failed:
        raise RuntimeError("Failed to parse FASTA file")

    print("trimming triplets")
    data = data[0:pos]
    row_ind = row_ind[0:pos]
    col_ind = col_ind[0:pos]
    if y_file:
        save_npz(y_file, coo_matrix(kmer_counts), compressed=False)

    print("building csr_matrix")
    xt = csr_matrix((data, (row_ind, col_ind)), shape=(M, n_cols), dtype=float_t)
    print("saving csr matrix to file = ", x_file)
    save_npz(x_file, xt, compressed=False)
