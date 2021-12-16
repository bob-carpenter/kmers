# kmerexpr

`kmerexpr` is a Python package for estimating isoform expression with
both the reads and transcriptomes represented as k-mer counts. 


## Installation

First ensure you're in a Python 3 virtual environment (see instructions below). 

To install the package and requirements:
```pip install -r kmerexpr/requirements.txt```

### Setup a virtual environment
Create a [Python 3 virtual environment](https://docs.python.org/3/tutorial/venv.html).

For Unix or MacOS this can be done by executing: `source activate [env name]`
1. `python3 -m venv kmer-env`
2. `source activate kmer-env/bin/activate`


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

## Experiment

The initial point of this repo is to measure how efficiently and
reliably we can use k-mers without alignment to estimate expression
for a single RNA-seq sample.  The steps required are as follows, all
of which are done other than the eval.

#### Step 1.  Transcriptome to x

* read transcriptome from Fasta format
* shred isoforms to k-mers
* count k-mers and normalize to stochastic matrix with prob of
k-mer given isoform
* build sparse matrix in Python
* serialize sparse matrix in scipy.sparse's .npz format

#### Step 2.  Simulate expression theta, reads y

* generate simplex `theta = softmax(alpha)` by generating where `alpha ~ Normal(0,
4)` to match the model, or just sample from a Dirichlet
* generate isoforms for reads `z[n] ~ categorical(theta)`
* read transcriptome from Fasta format
* for each read, generate a position uniformly along the sequence for
`z[n]`
* serialize read sequences to disk in Fasta (or Fastq) format
* generate `y` vector by shreadding reads and collecting counts
* serialize `y` to disk in whatever format Python uses for dense
vectors

#### Step 3. Instantiate model class

* Deserialize `x` stochastic matrix converting isoform expression
simplexes to k-mer expression simplexes
* Deserialize `y` vector of counts of k-mers
* Construct model class implementing size methods and a single method
to return log density and gradient

#### Step 4.  Optimize to estimate theta

* generate initial guess for `theta` (e.g., uniform)
* run optimizer for density as implemented in model class

#### Step 5.  Compare results

* we can look at difference between `theta` and its estimate
    * squared error
* if it looks good
    * bootstrap error bars for individual fit
    * iterate for different theta
	* cross-validate to predict held-out data


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

