import os
from kmerexpr.fasta import read_fasta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = os.path.join(HERE, "data")


def test_fasta1():
    ISO_FILE = os.path.join(DATA_PATH, "test1.fsa")
    with open(ISO_FILE, "r") as f:
        parser = read_fasta(f)
        greedy = list(parser)
        assert len(greedy) == 2
        assert "hello" in greedy[0].header
        assert len(greedy[1].sequence) == 5


def test_fasta_newlines():
    ISO_FILE = os.path.join(DATA_PATH, "test4.fsa")
    ISO_FILE_NL = os.path.join(DATA_PATH, "test5.fsa")

    with open(ISO_FILE, "r") as f1:
        with open(ISO_FILE_NL, "r") as f2:
            parser1 = read_fasta(f1)
            parser2 = read_fasta(f2)
            for s1, s2 in zip(parser1, parser2):
                assert s1.header == s2.header
                assert s1.sequence == s2.sequence
