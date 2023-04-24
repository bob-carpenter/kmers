#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

std::array<char, 256> init_base_ids() {
    std::array<char, 256> ids;
    std::fill_n(ids.begin(), 256, -1);

    ids['A'] = 0;
    ids['C'] = 1;
    ids['G'] = 2;
    ids['T'] = 3;

    return ids;
}

static std::array<char, 256> base_ids = init_base_ids();

uint32_t kmer_to_id(const char *kmer, int len) {
    uint32_t i = 0;
    uint32_t id = 0;
    for (int i = 0; i < len; ++i)
        id = id * 4 + base_ids[kmer[i]];

    return id;
}

bool valid_kmer(const char *kmer, int len) {
    for (int i = 0; i < len; ++i)
        if (base_ids[kmer[i]] < 0)
            return false;

    return true;
}

std::pair<std::unordered_map<uint32_t, uint32_t>, uint32_t> collect_kmers(const std::string &sequence, int klen) {
    std::unordered_map<uint32_t, uint32_t> kmer_counts;
    uint32_t total_count = 0;
    for (int i = 0; i < sequence.length() - klen + 1; ++i) {
        if (!valid_kmer(sequence.data() + i, klen))
            continue;
        else {
            auto id = kmer_to_id(sequence.data() + i, klen);
            auto kmer_count = kmer_counts.find(id);
            if (kmer_count != kmer_counts.end())
                kmer_count->second++;
            else
                kmer_counts[id] = 1;
            total_count++;
        }
    }
    return std::make_pair(std::move(kmer_counts), total_count);
}

std::pair<std::string, std::string> get_next_sequence(std::iostream &stream) {
    std::string line, header, sequence;

    getline(stream, header);
    auto pos = stream.tellg();
    while (getline(stream, line)) {
        if (line[0] == '>') {
            stream.seekg(pos);
            break;
        }
        while (line.back() == '\n' || line.back() == ' ')
            line.pop_back();
        sequence += line;
        pos = stream.tellg();
    }

    return std::make_pair(std::move(header), std::move(sequence));
}

extern "C" {
int fasta_to_kmers_sparse(const char *fname, int K, float *data, int *row_ind, int *col_ind, int *total_kmer_counts,
                          int max_size, int *n_elements, int *n_cols) {
    std::fstream stream;
    stream.open(fname, std::ios::in);
    if (!stream)
        return -1;

    int seqid = 0;
    int pos = 0;
    while (true) {
        auto [header, sequence] = get_next_sequence(stream);
        if (!header.length())
            break;
        if (header.find("PREDICTED") != std::string::npos)
            continue;

        auto [kmer_counts, total_count] = collect_kmers(sequence.data(), K);

        for (auto &[kmer, count] : kmer_counts) {
            if (pos >= max_size)
                return -1;
            data[pos] = float(count) / total_count;

            row_ind[pos] = kmer;
            col_ind[pos] = seqid;
            total_kmer_counts[kmer]++;
            pos++;
        }

        if (seqid % 10000 == 0)
            std::cout << seqid << " " << pos << std::endl;

        seqid++;
    }

    *n_elements = pos;
    *n_cols = seqid;
    return 0;
}
}
