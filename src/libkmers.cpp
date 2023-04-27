#include "libkmers.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

class ThreadPool {
  public:
    ThreadPool() { Start(); }
    ~ThreadPool() { Stop(); }

    void Start() {
        const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
        threads.resize(num_threads);
        for (uint32_t i = 0; i < num_threads; i++)
            threads.at(i) = std::thread([this]() { this->ThreadLoop(); });
    }
    void QueueJob(const std::function<void()> &job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }
    void Stop() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread &active_thread : threads) {
            active_thread.join();
        }
        threads.clear();
    };


    int n_jobs() const { return jobs.size(); }

  private:
    void ThreadLoop() {
        while (true) {
            std::function<void()> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                mutex_condition.wait(lock, [this] { return !jobs.empty() || should_terminate; });
                if (should_terminate)
                    return;

                job = jobs.front();
                jobs.pop();
            }
            job();
        }
    }

    bool should_terminate = false;
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::queue<std::function<void()>> jobs;
};

std::array<char, 256> init_base_ids() {
    std::array<char, 256> ids;
    std::fill_n(ids.begin(), 256, -1);

    ids['A'] = 0;
    ids['C'] = 1;
    ids['G'] = 2;
    ids['T'] = 3;

    return ids;
}

static const std::array<char, 256> base_ids = init_base_ids();

inline void atomic_increment(int *arr, int index) {
    std::atomic_ref<int> el(arr[index]);
    el.fetch_add(1, std::memory_order_relaxed);
}

std::unordered_map<uint32_t, uint32_t> collect_kmers(const std::string &sequence, int klen) {
    std::unordered_map<uint32_t, uint32_t> kmer_counts;
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
        }
    }
    return std::move(kmer_counts);
}

void fill_indices(int K, int seq_total_count, int seqid, const std::string &sequence, float *data, uint64_t *row_ind,
                  uint64_t *col_ind, int *total_kmer_counts) {
    const auto kmer_counts = collect_kmers(sequence.data(), K);
    uint64_t i = 0;
    for (auto &[kmer, count] : kmer_counts) {
        data[i] = float(count) / seq_total_count;
        row_ind[i] = kmer;
        col_ind[i] = seqid;
        atomic_increment(total_kmer_counts, kmer);
        i++;
    }
}

std::pair<std::string, std::string> get_next_sequence_fasta(std::iostream &stream) {
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

inline int max_total_subseq(int L, int K) { return (L < K) ? 0 : L - K + 1; }

uint64_t remove_invalid_elements(uint64_t len, float *data, uint64_t *row_ind, uint64_t *col_ind) {
    uint64_t nnz = 0;
    for (uint64_t i = 0; i < len; ++i) {
        if (col_ind[i] != std::numeric_limits<uint64_t>::max()) {
            row_ind[nnz] = row_ind[i];
            col_ind[nnz] = col_ind[i];
            data[nnz] = data[i];
            nnz++;
        }
    }
    return nnz;
}

std::string get_sequence_fastq(const std::string &fname) {
    std::fstream stream;
    stream.open(fname, std::ios::in);
    if (!stream)
        return "";

    std::string line, sequence;
    while (true) {
        getline(stream, line);
        if (!line.length() || line[0] != '@')
            break;
        getline(stream, line);
        sequence += line.substr(0, line.length() - 1);
        getline(stream, line);
        if (!line.length() || line[0] != '+')
            break;

        getline(stream, line);
    }

    return sequence;
}

extern "C" {
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

int fasta_to_kmers_sparse(int n_files, const char *fnames[], int K, float *data, uint64_t *row_ind, uint64_t *col_ind,
                          int *total_kmer_counts, uint64_t max_size, uint64_t *nnz, int *n_cols) {
    std::fill(col_ind, col_ind + max_size, -1);

    int seqid = 0;
    uint64_t pos = 0;
    ThreadPool pool;
    for (int i_file; i_file < n_files; ++i_file) {
        std::fstream stream;
        stream.open(fnames[i_file], std::ios::in);
        if (!stream)
            return -1;
        printf("processing %s\n", fnames[i_file]);

        while (true) {
            std::string header, sequence;
            std::tie(header, sequence) = get_next_sequence_fasta(stream);
            if (!header.length())
                break;
            if (header.find("PREDICTED") != std::string::npos)
                continue;

            const int n_sub_seq = max_total_subseq(sequence.length(), K);
            pool.QueueJob([K, n_sub_seq, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
                fill_indices(K, n_sub_seq, seqid, sequence, data + pos, row_ind + pos, col_ind + pos,
                             total_kmer_counts);
            });

            pos += n_sub_seq;
            if (pos > max_size) {
                pool.Stop();
                return -2;
            }

            seqid++;
        }
    }
    pool.Stop();

    *nnz = remove_invalid_elements(pos, data, row_ind, col_ind);
    *n_cols = seqid;
    return 0;
}

int fastq_to_kmers_sparse(int n_files, const char *fname[], int K, float *data, uint64_t *row_ind, uint64_t *col_ind,
                          int *total_kmer_counts, uint64_t max_size, uint64_t *n_elements, int *n_cols) {
    std::fill(col_ind, col_ind + max_size, std::numeric_limits<uint64_t>::max());

    uint64_t pos = 0;
    ThreadPool pool;
    for (int seqid = 0; seqid < n_files; ++seqid) {
        printf("processing %s\n", fname[seqid]);
        std::string sequence = get_sequence_fastq(fname[seqid]);
        const int n_sub_seq = max_total_subseq(sequence.length(), K);

        pool.QueueJob([K, n_sub_seq, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
            fill_indices(K, n_sub_seq, seqid, sequence, data + pos, row_ind + pos, col_ind + pos, total_kmer_counts);
        });

        pos += n_sub_seq;
    }

    pool.Stop();
    pos = remove_invalid_elements(pos, data, row_ind, col_ind);

    *n_elements = pos;
    *n_cols = n_files;
    return 0;
}
}
