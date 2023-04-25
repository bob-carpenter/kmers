#ifndef LIBKMERS_H
#define LIBKMERS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t kmer_to_id(const char *kmer, int len);
bool valid_kmer(const char *kmer, int len);
int fasta_to_kmers_sparse(const char *fname, int K, float *data, int *row_ind, int *col_ind, int *total_kmer_counts,
                          int max_size, int *n_elements, int *n_cols);

#ifdef __cplusplus
}
#endif

    
#endif
