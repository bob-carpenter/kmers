from kmerexpr.simulate_reads import simulate_reads
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test1():
    ISO_FILE = os.path.join(ROOT, "data", "GRCh38_latest_rna.fna")
    RNA_SEQ_FILE = os.path.join(ROOT, "data", "rna_seq_sim.fna")
    N = 10_000
    L = 50
    simulate_reads(ISO_FILE, RNA_SEQ_FILE, N, L)
    # BMW: Test is a no op?
    assert 1 == 1
