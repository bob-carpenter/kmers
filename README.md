# K-mers

This is a package for doing gene expression with k-mers. The basic
code is in a top-level file `read.py` for now.

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

It assembles the transcriptome as a sparse matrix then performs a
sample matrix-vector multiply.

## Documentation

There is some documentation for the model that we are going to fit in
the top level file `model.tex`.  To build that, do

```
> cd kmers
> pdflatex model.tex
> open model.pdf
```
