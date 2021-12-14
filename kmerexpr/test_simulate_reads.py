import numpy as np
from simulate_reads import simulate_reads
import pytest

def test1():
    ISO_FILE ='data/GRCh38_latest_rna.fna'
    RNA_SEQ_FILE = 'data/rna_seq_sim.fna'
    N = 10_000
    L = 50
    simulate_reads(ISO_FILE, RNA_SEQ_FILE, N, L)
    assert 1 == 1
