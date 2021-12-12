# kmerexpr

`kmerexpr` is a Python package for estimating isoform expression with
both the reads and transcriptomes represented as k-mer counts. 

## Data download

The fasta format of the human transcriptome is available as one of the
human gene resources at NCBI.  Here's the download page.

*
[NCBI's Human Genome Resources](https://www.ncbi.nlm.nih.gov/projects/genome/guide/human/index.shtml)

You want to download the "Fasta" format of "Refseq Transcripts".  As
of now the latest version is GRCh38.  Place that in a directory called

* `/data`

and you should be good to execute the `read.py` program from the shell
as

```
> cd kmers
> python3 read.py
```

It assembles the transcriptome as a sparse matrix and then serializes
it to a npz file.  


## Documentation

There is some documentation for the model that we are going to fit in
the top level file `model.tex`.  To build the pdf, do

```
> cd kmers
> pdflatex model.tex
> open model.pdf
```

## Licensing

The code in this repo is BSD-3 licensed and the doc is CC-BY ND 3.0
licensed.


## Dependencies

### `scipy` / `numpy`

Scipy and Numpy are BSD-3 licensed.

### `fastaparser`

This is GPLv3 licensed. If this approach works out, we'll replace that
with a non-copylefted Fasta parser as the functionality we need is not
that difficult to code.

