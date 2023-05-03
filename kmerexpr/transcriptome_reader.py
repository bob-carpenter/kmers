from scipy.sparse import csr_matrix, save_npz, load_npz  # BSD-3
import numpy as np  # BSD-3
from kmerexpr.libkmers import fasta_to_kmers_csr


def load_xy(x_file, y_file):
    if isinstance(x_file, csr_matrix):
        x = x_file
    else:
        x = load_npz(x_file)

    if isinstance(y_file, np.ndarray):
        y = y_file
    else:
        y = load_npz(y_file).toarray().squeeze()
    return x, y


def transcriptome_to_x(K: str,
                       fasta_files: list[str],
                       x_file: str = "",
                       L: int = 0,
                       max_nz: int =0,
                       float_t: type = np.float32,
                       concatenate_subseq: bool = False,
                       ):
    if float_t != np.float32:
        raise ValueError("Only float_t=np.float32 currently supported for fast reader")

    M = 4**K
    data, row_ind, col_ind, n_cols = \
        fasta_to_kmers_csr(fasta_files, K, max_nz, L=L, concatenate_subseq=concatenate_subseq)

    print("converting CSR matrix to scipy form")
    xt = csr_matrix((data, col_ind, row_ind), shape=(M, n_cols), dtype=float_t)

    if x_file:
        print("saving csr matrix to file = ", x_file)
        save_npz(x_file, xt, compressed=False)

    return xt
