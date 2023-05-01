#include "libkmers.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

using list_of_lists_t = std::vector<std::vector<std::pair<uint64_t, float>>>;
using coo_tup = std::tuple<uint64_t, uint64_t, float>;
using list_of_coo = std::vector<coo_tup>;

template <typename T>
class ThreadPoolLocalData {
  public:
    ThreadPoolLocalData() { start(); }
    ~ThreadPoolLocalData() { stop(); }

    void start() {
        const char *libkmer_n_threads = getenv("KMER_NUM_THREADS");
        const uint32_t num_threads = (libkmer_n_threads == nullptr)
                                         ? std::thread::hardware_concurrency()
                                         : atoi(libkmer_n_threads); // Max # of threads the system supports
        threads.resize(num_threads);
        localdata.resize(num_threads);
        for (uint32_t i = 0; i < num_threads; i++) {
            threads.at(i) = std::thread([this, i]() { this->ThreadLoop(localdata.at(i)); });
        }
    }
    void queue_job(const std::function<void(T &)> &job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }
    void stop() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread &active_thread : threads) {
            active_thread.join();
        }
        threads.clear();
    }

    list_of_lists_t reduce(const uint64_t nrows) {
        std::cout << "Reducing intermediate COO -> CSR\n";
        list_of_lists_t res(nrows);
        for (int i = 0; i < localdata.size(); ++i) {
            std::cout << "Joining " << i + 1 << " of " << localdata.size() << std::endl;

            for (auto &[row, col, val] : localdata[i])
                res[row].push_back(std::make_pair(col, val));

            localdata[i] = {};
        }

        uint64_t n_update = nrows / 20;
        for (uint64_t i = 0; i < nrows; ++i) {
            if (i % n_update == 0)
                std::cout << "Sorting row " << i << " of " << nrows << std::endl;

            auto &row = res[i];
            std::sort(row.begin(), row.end());
        }
        return std::move(res);
    }

    int n_jobs() const { return jobs.size(); }

  private:
    void ThreadLoop(T &local_data) {
        while (true) {
            std::function<void(T &)> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                mutex_condition.wait(lock, [this] { return !jobs.empty() || should_terminate; });
                if (should_terminate)
                    return;

                job = jobs.front();
                jobs.pop();
            }
            job(local_data);
        }
    }

    bool should_terminate = false;
    std::vector<T> localdata;
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::queue<std::function<void(T &)>> jobs;
};

class ThreadPool {
  public:
    ThreadPool() { start(); }
    ~ThreadPool() { stop(); }

    void start() {
        const uint32_t num_threads = std::thread::hardware_concurrency(); // Max # of threads the system supports
        threads.resize(num_threads);
        for (uint32_t i = 0; i < num_threads; i++)
            threads.at(i) = std::thread([this]() { this->ThreadLoop(); });
    }
    void queue_job(const std::function<void()> &job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }
    void stop() {
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

void fill_indices_coo(int K, int seq_total_count, int seqid, const std::string &sequence, float *data,
                      uint64_t *row_ind, uint64_t *col_ind, int *total_kmer_counts) {
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

void fill_indices_csr(list_of_coo &lil, int K, int seq_total_count, int seqid, const std::string &sequence) {
    const auto kmer_counts = collect_kmers(sequence.data(), K);
    for (auto &[kmer, count] : kmer_counts) {
        const float data = float(count) / seq_total_count;
        lil.push_back(std::make_tuple(kmer, seqid, data));
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

inline int max_total_kmers(int L, int K) { return (L < K) ? 0 : L - K + 1; }

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

int fasta_to_kmers_sparse_cat_subseq(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                                     uint64_t *col_ind, int *total_kmer_counts, uint64_t max_size, uint64_t *nnz,
                                     int *n_cols) {
    std::fill(col_ind, col_ind + max_size, -1);

    uint64_t pos = 0;
    ThreadPool pool;
    for (int seqid; seqid < n_files; ++seqid) {
        std::fstream stream;
        stream.open(fnames[seqid], std::ios::in);
        if (!stream)
            return -1;
        printf("processing %s\n", fnames[seqid]);

        std::string sequence;
        while (true) {
            std::string header, subseq;
            std::tie(header, subseq) = get_next_sequence_fasta(stream);
            if (!header.length())
                break;
            if (header.find("PREDICTED") != std::string::npos)
                continue;
            sequence += subseq;
        }
        if (sequence.length() < L)
            continue;

        const int max_kmers = max_total_kmers(sequence.length(), K);
        pool.queue_job([K, max_kmers, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
            fill_indices_coo(K, max_kmers, seqid, sequence, data + pos, row_ind + pos, col_ind + pos,
                             total_kmer_counts);
        });

        pos += max_kmers;
        if (pos > max_size) {
            pool.stop();
            return -2;
        }
    }
    pool.stop();

    *nnz = remove_invalid_elements(pos, data, row_ind, col_ind);
    *n_cols = n_files;
    return 0;
}

int fasta_to_kmers_csr_cat_subseq(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                                  uint64_t *col_ind, int *total_kmer_counts, uint64_t max_size, uint64_t *nnz,
                                  int *n_cols) {

    uint64_t pos = 0;
    ThreadPoolLocalData<list_of_coo> pool;
    for (int seqid; seqid < n_files; ++seqid) {
        std::fstream stream;
        stream.open(fnames[seqid], std::ios::in);
        if (!stream)
            return -1;
        printf("processing %s\n", fnames[seqid]);

        std::string sequence;
        while (true) {
            std::string header, subseq;
            std::tie(header, subseq) = get_next_sequence_fasta(stream);
            if (!header.length())
                break;
            if (header.find("PREDICTED") != std::string::npos)
                continue;
            sequence += subseq;
        }
        if (sequence.length() < L)
            continue;

        const int max_kmers = max_total_kmers(sequence.length(), K);
        pool.queue_job([K, max_kmers, seqid, sequence](list_of_coo &lil) {
            fill_indices_csr(lil, K, max_kmers, seqid, sequence);
        });

        pos += max_kmers;
        if (pos > max_size) {
            pool.stop();
            return -2;
        }
    }
    pool.stop();

    pos = 0;
    uint64_t M = std::pow(4, K);

    auto lil_mat = pool.reduce(M);
    for (uint64_t i = 0; i < M; ++i) {
        auto &row = lil_mat[i];

        total_kmer_counts[i] = row.size();
        row_ind[i] = pos;
        for (auto &[col, val] : row) {
            col_ind[pos] = col;
            data[pos] = val;
            pos++;
        }
    }

    row_ind[M] = *nnz = pos;
    *n_cols = n_files;
    return 0;
}

int fasta_to_kmers_csr(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                       uint64_t *col_ind, int *total_kmer_counts, uint64_t max_size, uint64_t *nnz, int *n_cols) {
    int seqid = 0;
    uint64_t pos = 0;
    ThreadPoolLocalData<list_of_coo> pool;
    for (int i_file = 0; i_file < n_files; ++i_file) {
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
            if (sequence.length() < L)
                continue;

            const int max_kmers = max_total_kmers(sequence.length(), K);
            pool.queue_job([K, max_kmers, seqid, sequence](list_of_coo &lil) {
                fill_indices_csr(lil, K, max_kmers, seqid, sequence);
            });

            pos += max_kmers;
            if (pos > max_size) {
                pool.stop();
                return -2;
            }

            seqid++;
        }
    }
    pool.stop();

    pos = 0;
    uint64_t M = std::pow(4, K);

    auto lil_mat = pool.reduce(M);
    for (uint64_t i = 0; i < M; ++i) {
        auto &list = lil_mat[i];

        total_kmer_counts[i] = list.size();
        row_ind[i] = pos;
        for (auto &[col, val] : list) {
            col_ind[pos] = col;
            data[pos] = val;
            pos++;
        }
    }

    row_ind[M] = *nnz = pos;
    *n_cols = seqid;
    return 0;
}

int fasta_to_kmers_sparse(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                          uint64_t *col_ind, int *total_kmer_counts, uint64_t max_size, uint64_t *nnz, int *n_cols) {
    std::fill(col_ind, col_ind + max_size, -1);

    int seqid = 0;
    uint64_t pos = 0;
    ThreadPool pool;
    for (int i_file = 0; i_file < n_files; ++i_file) {
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
            if (sequence.length() < L)
                continue;

            const int max_kmers = max_total_kmers(sequence.length(), K);
            pool.queue_job([K, max_kmers, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
                fill_indices_coo(K, max_kmers, seqid, sequence, data + pos, row_ind + pos, col_ind + pos,
                                 total_kmer_counts);
            });

            pos += max_kmers;
            if (pos > max_size) {
                pool.stop();
                return -2;
            }

            seqid++;
        }
    }
    pool.stop();

    *nnz = remove_invalid_elements(pos, data, row_ind, col_ind);
    *n_cols = seqid;
    return 0;
}

int fastq_to_kmers_sparse(int n_files, const char *fname[], int K, int L, float *data, uint64_t *row_ind,
                          uint64_t *col_ind, int *total_kmer_counts, uint64_t max_size, uint64_t *n_elements,
                          int *n_cols) {
    std::fill(col_ind, col_ind + max_size, std::numeric_limits<uint64_t>::max());

    uint64_t pos = 0;
    ThreadPool pool;
    for (int seqid = 0; seqid < n_files; ++seqid) {
        printf("processing %s\n", fname[seqid]);
        std::string sequence = get_sequence_fastq(fname[seqid]);
        const int max_kmers = max_total_kmers(sequence.length(), K);

        if (sequence.length() < L)
            continue;

        pool.queue_job([K, max_kmers, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
            fill_indices_coo(K, max_kmers, seqid, sequence, data + pos, row_ind + pos, col_ind + pos,
                             total_kmer_counts);
        });

        pos += max_kmers;
    }

    pool.stop();
    pos = remove_invalid_elements(pos, data, row_ind, col_ind);

    *n_elements = pos;
    *n_cols = n_files;
    return 0;
}
}
