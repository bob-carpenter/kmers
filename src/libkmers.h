#ifndef LIBKMERS_H
#define LIBKMERS_H


#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#include <stdint.h>
#include <stdbool.h>
#endif

uint32_t kmer_to_id(const char *kmer, int len);
bool valid_kmer(const char *kmer, int len);
int fasta_count_kmers(int n_files, const char *fnames[], int K, int *total_kmer_counts);
int fasta_to_kmers_csr_cat_subseq(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                                  uint64_t *col_ind, uint64_t max_size, uint64_t *nnz, int *n_cols);
int fasta_to_kmers_csr(int n_files, const char *fname[], int K, int L, float *data, uint64_t *row_ind,
                       uint64_t *col_ind, uint64_t max_size, uint64_t *n_elements, int *n_cols);

#ifdef __cplusplus
}
#endif
#endif
