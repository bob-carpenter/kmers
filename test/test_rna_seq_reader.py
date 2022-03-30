from kmerexpr.rna_seq_reader import reads_to_y
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def test1():
    RNA_SEQ_FILE = os.path.join(ROOT, 'data',"rna_seq_sim.fna")
    print("RNA_SEQ_FILE =", RNA_SEQ_FILE)
    K = 5
    print("K =", K)
    y = reads_to_y(K, RNA_SEQ_FILE)
    print("y[0:10] =", y[0:10])
    print("y[(4**K - 10):(4**K)] =", y[(4 ** K - 10) : (4 ** K)])
    assert 1 == 1
