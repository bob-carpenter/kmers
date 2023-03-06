from kmerexpr.simulate_reads import simulate_reads
from kmerexpr.utils import Problem
def test1():
    file_name = "GRCh38_latest_rna.fna"
    N = 10_000
    L = 50
    problem = Problem(filename=file_name, K=8, N =N, L=L)
    simulate_reads(problem)
    # BMW: Test is a no op?
    assert 1 == 1