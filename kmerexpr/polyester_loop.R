library(Biostrings)
library(polyester)
library(seqinr)
library(readr)
library(stringr)
library(dplyr)
library(fs)
library(ggplot2)

dir.create("fastas")
duds <- file("duds.txt")

fastaWriter <- function(fasta, outputfile) {
  for (rowNum in 1:nrow(fasta)){
    seq_name = as.character(paste(">", fasta[rowNum,"seq_name"], sep = ""))
    write_lines(seq_name, outputfile, append = T)
    sequence = as.character(fasta[rowNum,"sequence"])
    write_lines(sequence, outputfile, append = T)
  }
}

master_fasta = readDNAStringSet("GRCh38_latest_rna.fna")

seq_name = names(master_fasta)
sequence = paste(master_fasta)

fasta_frame <- data.frame(seq_name, sequence)
fasta_frame$length <- nchar(fasta_frame$sequence)

# If there is any filtering you'd like to do, this is the stage at which these variables should be modified.
fasta_frame <- fasta_frame |>
  filter(length > 100) |>
  filter(str_detect(seq_name, "NM") | str_detect(seq_name, "XM"))

# An original version was very poorly performant, but readr is much faster.
fastaWriter(fasta_frame, "cleaned_fasta.fa")

fasta <- readDNAStringSet("cleaned_fasta.fa")

for (i in 1:length(fasta)) {
  seq_name = names(fasta[i])
  fn = str_split_fixed(seq_name, pattern = " ", n = 2)[1]
  sequence = paste(fasta[i])
  frame <- data.frame(seq_name, sequence)
  if (!(file.exists(file.path("fastas", paste(fn, ".fa", sep = ""))))) {
    fastaWriter(frame, file.path("fastas", paste(fn, ".fa", sep = "")))
  }
}

fold_changes <-matrix(c(1), nrow = 1, ncol = 1, byrow = T)

fastas <- list.files("fastas")

for (fa in fastas) {
  transcript <- readDNAStringSet(file.path("fastas", fa))
  readspertx = round(20 * width(transcript) / 100)
  size = as.numeric(readspertx * fold_changes / 4)
  print(fa)
  tryCatch({
    simulate_experiment(file.path("fastas", fa), reads_per_transcript = readspertx,
                        num_reps = c(1), fold_changes = fold_changes,
                        outdir = file.path("simulated_reads", path_ext_remove(fa)),
                        error_rate = 0.0001, size = size,
                        paired = T, bias = "cdnaf", gcbias = 2)
  }, error = function(e) {write_lines(fa, duds, append = T)
  })
}

# Concatenate all of the output FASTAs into a single FASTA file.

system('find . -type f -name "sample_01_1.fasta" -print0 |
       xargs -0 cat > combined_1.fa')

system('find . -type f -name "sample_01_2.fasta" -print0 |
       xargs -0 cat > combined_2.fa')

