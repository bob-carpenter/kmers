from kmerexpr.simulate_reads import simulate_reads
def test1():
    file_name = "GRCh38_latest_rna.fna"
    N = 10_000
    L = 50
    simulate_reads(file_name, N, L)
    # BMW: Test is a no op?
    assert 1 == 1

## old version
# def test1():
#     ISO_FILE = os.path.join(ROOT, "data", "GRCh38_latest_rna.fna")
#     RNA_SEQ_FILE = os.path.join(ROOT, "data", "rna_seq_sim.fna")
#     THETA_FILE = os.path.join(ROOT, "data", "theta")
#     N = 10_000
#     L = 50
#     simulate_reads(ISO_FILE, RNA_SEQ_FILE, THETA_FILE, N, L)
#     # BMW: Test is a no op?
#     assert 1 == 1
