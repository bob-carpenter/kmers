# Connect salmon output to polyester input to create simulated samples.
# At a high level, we connect the salmon output (quant.sf.gz) data 
# to the readmat variable of the simulate_experiment_countmat function.

# Load required packages
library(polyester)
library(Biostrings)

# Set paths to input files
fasta_file <- "data/GRCh38_latest_rna.fna.gz"
salmon_output_file <- "ERR204899/quant.sf.gz"

# We want to connect the NUMREADS or TPM from the salmon output
# to our quant_mat. Note that we need the number of rows in our quant_mat to match the number
# of transcripts in the GTF file

# Read Salmon output file
quants <- read.delim(salmon_output_file, header = TRUE, stringsAsFactors = FALSE)
salmon_mat <- as.matrix(quants)
fasta_transcript_names = names(Biostrings::readDNAStringSet(fasta_file))
quant_mat_nrows = length(fasta_transcript_names)

# Create zeroed-out quant_mat with the correct dimensions
quant_mat = as.matrix(rep(0,quant_mat_nrows))

# Shorten transcript names in the fasta file to only the transcript id.

# Uncomment this line if the reference transcripts have a '|' separator in the names
# new_rownames <- sapply(strsplit(fasta_transcript_names, "\\|"), "[[", 1)
# Uncomment this line if the reference transcripts have a ' ' separator in the names

new_rownames <- sapply(strsplit(fasta_transcript_names, "\\ "), "[[", 1)

rownames(quant_mat) <- new_rownames

# Verify that the transcript ids from quant.sf.gz are all in our 
# list of transcripts from GRCh38_latest_rna.fna
stopifnot(all(salmon_mat[,1] %in% rownames(quant_mat))) 

# Find indexess of quant_mat where its rowname matches first column of salmon_mat
idx <- match(salmon_mat[,1], rownames(quant_mat), nomatch = 0)

# This is a good place to scale or round the NumReads.
# This is also where we change to use TPM instead of NumReads.
quant_mat[idx, 1] <- as.numeric(salmon_mat[, 5])

simulated_counts <- simulate_experiment_countmat(fasta = fasta_file,
                                                 readmat = quant_mat,
                                                 paired = FALSE,
                                                 reads_per_transcript = 200,
                                                 readlen = 100,
                                                 error_rate = 0.0001,
                                                 bias = "cdnaf",
                                                 gcbias = 2,
                                                 seed = 42)
