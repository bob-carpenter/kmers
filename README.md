# kmerexpr

`kmerexpr` is a Python package for estimating isoform expression with
both the reads and transcriptomes represented as k-mer counts. 

## Basic usage

Given a fasta formatted transcriptome in `tr_file.fasta` the way to
create a model is

```python
import transcriptome_reader as tr
import multinomial_model as mm

K = 8  # size of k-mers to use
ISO_FILE = <path to transcriptome file in fasta format>
X_FILE = <path to save isoform to kmer matrix>
tr.transcriptome_to_x(K, ISO_FILE, X_FILE)
y = <vector of observed reads>
model = mm.multinomial_model(X_FILE, y)
theta = <simplex of isoform expression>
logp, grad = model.logp_grad(theta)
```

The resulting value is a tuple with `logp` assigned the value of the
log probability density function at the simplex `theta` and `grad`
assigned the gradient at `theta`.


## Data download

The fasta format of the human transcriptome is available as one of the
human gene resources at NCBI.  Here's the download page.

* [NCBI's Human Genome Resources](https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml)

You want to download the "Fasta" format of "Refseq Transcripts".  As
of now the latest version is GRCh38.  Place that in a directory called

* `/data`

and you should be good to execute all of the tests or use that for the
isoform file.


## Documentation

There is some documentation for the model that we are going to fit in
the top level file `model.tex`.  To build the pdf, do

```console
> cd kmers
> pdflatex model.tex
> open model.pdf
```

## Unit testing

The `pytest`-based unit tests may be run from the top level as
follows. 

```console
> cd kmers
> pytest -s
```

The `-s` option provides streaming output of any print statements
within the code, which provide updates on progress of long-running
processes. It's also possible to test a single function, e.g.,

```console
> cd kmers
> pytest -s kmerexpr/test_multinomial_model.py
```

This tests the fasta reader along with the serializer and deserializer
for the matrix converting a simplex over isoforms to a simplex over
k-mers.


## Licensing

The code in this repo is BSD-3 licensed and the doc is CC-BY ND 3.0
licensed.


## Dependencies

### `scipy` / `numpy`

Scipy and Numpy are BSD-3 licensed.

### `fastaparser`

The fastaparser package is GPLv3 licensed. If this approach works out,
we'll replace that with a non-copylefted Fasta parser as the
functionality we need is not that difficult to code.

