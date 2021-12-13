import numpy as np
from rna_seq_reader import reads_to_y
import pytest

def test1():
    RNA_SEQ_FILE = 'rna_seq_sim.fna'
    K = 5
    y = reads_to_y(K, RNA_SEQ_FILE)
