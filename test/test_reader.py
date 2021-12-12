import numpy as np
from scipy.sparse import load_npz
import kmerexpr as ke

K = 2
TRANSCRIPTOME_FILE='data/test1.fsa'
X_FILE = f"data/xt1_{K}.npz"

def test_answer():
    ke.transcriptome_to_x(K, TRANSCRIPTOME_FILE, X_FILE)
    assert 1 == 1
