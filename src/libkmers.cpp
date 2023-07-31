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

using lil_entry = std::pair<uint64_t, float>;
using coo_entry = struct {
    uint64_t row;
    uint64_t col;
    float data;
};
using list_of_coo = std::vector<coo_entry>;
using kmer_count_t = std::unordered_map<uint64_t, uint32_t>;

struct Timer {
    struct timespec ts, tf;

    Timer() { start(); }
    void start() { clock_gettime(CLOCK_MONOTONIC, &ts); }
    void stop() { clock_gettime(CLOCK_MONOTONIC, &tf); }
    double elapsed() { return (tf.tv_sec - ts.tv_sec) + (tf.tv_nsec - ts.tv_nsec) * 1E-9; }
};

template <typename T>
class ThreadPoolLocalData {
  public:
    ThreadPoolLocalData() { start(); }
    ~ThreadPoolLocalData() { finalize(); }

    void start() {
        const char *libkmer_n_threads = getenv("KMER_NUM_THREADS");
        num_threads_ = (libkmer_n_threads == nullptr) ? std::thread::hardware_concurrency()
                                                      : atoi(libkmer_n_threads); // Max # of threads the system supports
        threads.resize(num_threads_);
        localdata.resize(num_threads_);
        for (uint32_t i = 0; i < num_threads_; i++)
            threads.at(i) = std::thread([this, i]() { this->ThreadLoop(localdata.at(i)); });
    }
    void queue_job(const std::function<void(T &)> &job) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            jobs.push(job);
        }
        mutex_condition.notify_one();
    }
    void finalize() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            should_terminate = true;
        }
        mutex_condition.notify_all();
        for (std::thread &thread : threads)
            thread.join();

        threads.clear();
    }

    void to_csr(uint64_t nrows, uint64_t *row_ind, uint64_t *col_ind, float *data) {
        std::cout << "Converting intermediate COO -> CSR\n";

        threads.resize(num_threads_);
        Timer timer;
        const uint64_t chunk_size = nrows / num_threads_;
        // Pre-sort COO matrices for each thread
        for (int i_thr = 0; i_thr < num_threads_; ++i_thr)
            threads[i_thr] = std::thread([this, i_thr] {
                std::sort(localdata[i_thr].begin(), localdata[i_thr].end(),
                          [](const auto &a, const auto &b) { return a.row < b.row; });
            });
        for (auto &thread : threads)
            thread.join();

        timer.stop();
        std::cout << "Pre-sort took " << timer.elapsed() << " seconds\n";

        timer.start();
        std::vector<std::vector<uint64_t>> row_start_ptrs(num_threads_);
        for (auto &vec : row_start_ptrs)
            vec.resize(num_threads_ + 1);
        std::vector<uint64_t> data_offsets(num_threads_ + 1);

        for (int i_chunk = 0; i_chunk < num_threads_; ++i_chunk) {
            threads[i_chunk] = std::thread([this, i_chunk, chunk_size, nrows, &row_start_ptrs, &data_offsets] {
                uint64_t row_low = i_chunk * chunk_size;
                uint64_t row_high = (i_chunk + 1) * chunk_size;
                if (i_chunk == num_threads_ - 1)
                    row_high = nrows;

                for (int i_thr = 0; i_thr < num_threads_; ++i_thr) {
                    auto &ld = localdata[i_thr];
                    uint64_t row_idx = row_start_ptrs[i_thr][i_chunk];
                    while (row_idx < ld.size() && ld[row_idx].row < row_high)
                        row_idx++;
                    row_start_ptrs[i_thr][i_chunk + 1] = row_idx;

                    data_offsets[i_chunk + 1] += row_idx;
                }
            });
        }
        for (auto &thread : threads)
            thread.join();

        timer.stop();
        std::cout << "Indexing took " << timer.elapsed() << " seconds\n";

        timer.start();
        for (int i_chunk = 0; i_chunk < num_threads_; ++i_chunk) {
            threads[i_chunk] = std::thread(
                [this, i_chunk, chunk_size, &row_start_ptrs, &data_offsets, row_ind, col_ind, data, nrows]() {
                    uint64_t pos = data_offsets[i_chunk];
                    std::vector<lil_entry> rowvec;
                    std::vector<uint64_t> row_ptrs(num_threads_);
                    for (int i = 0; i < num_threads_; ++i)
                        row_ptrs[i] = row_start_ptrs[i][i_chunk];

                    uint64_t row_low = i_chunk * chunk_size;
                    uint64_t row_high = (i_chunk + 1) * chunk_size;
                    if (i_chunk == num_threads_ - 1)
                        row_high = nrows;

                    for (uint64_t i_row = row_low; i_row < row_high; ++i_row) {
                        rowvec.clear();
                        for (int i = 0; i < localdata.size(); ++i) {
                            auto &ld = localdata[i];
                            while (row_ptrs[i] < ld.size() && ld[row_ptrs[i]].row == i_row) {
                                auto &entry = ld[row_ptrs[i]];
                                rowvec.push_back({entry.col, entry.data});
                                row_ptrs[i]++;
                            }
                        }
                        std::sort(rowvec.begin(), rowvec.end(),
                                  [](const auto &a, const auto &b) { return a.first < b.first; });

                        row_ind[i_row] = pos;
                        for (auto &[col, val] : rowvec) {
                            col_ind[pos] = col;
                            data[pos] = val;
                            pos++;
                        }
                    }
                });
        }
        for (auto &thread : threads)
            thread.join();
        threads.clear();
        timer.stop();
        std::cout << "Finalizing took " << timer.elapsed() << " seconds\n";

        row_ind[nrows] = data_offsets[num_threads_];
    }

    int n_jobs() const { return jobs.size(); }

  private:
    void ThreadLoop(T &local_data) {
        while (true) {
            std::function<void(T &)> job;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                mutex_condition.wait(lock, [this] { return !jobs.empty() || should_terminate; });
                if (jobs.empty() && should_terminate)
                    return;

                job = jobs.front();
                jobs.pop();
            }
            job(local_data);
        }
    }

    uint32_t num_threads_;
    bool should_terminate = false;
    std::vector<T> localdata;
    std::vector<std::thread> threads;
    std::mutex queue_mutex;
    std::condition_variable mutex_condition;
    std::queue<std::function<void(T &)>> jobs;
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

std::pair<kmer_count_t, uint32_t> collect_kmers(const std::string &sequence, int klen) {
    kmer_count_t kmer_counts;
    uint32_t total_count = 0;
    kmer_counts.reserve(sequence.length() - klen + 1);
    for (int i = 0; i < sequence.length() - klen + 1; ++i) {
        if (!valid_kmer(sequence.data() + i, klen))
            continue;
        auto id = kmer_to_id(sequence.data() + i, klen);
        ++kmer_counts[id];
        ++total_count;
    }
    return std::make_pair(std::move(kmer_counts), total_count);
}

void fill_indices_coo(list_of_coo &coo, int K, uint64_t seqid, const std::string &sequence) {
    const auto [kmer_counts, total_count] = collect_kmers(sequence.data(), K);
    for (auto &[kmer, count] : kmer_counts) {
        const float data = float(count) / total_count;
        coo.emplace_back(coo_entry{kmer, seqid, data});
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

void count_kmers(int K, const std::string &sequence, int *kmer_counts) {
    for (int i = 0; i < sequence.length() - K + 1; ++i) {
        if (!valid_kmer(sequence.data() + i, K))
            continue;

        auto id = kmer_to_id(sequence.data() + i, K);
        atomic_increment(kmer_counts, id);
    }
}

extern "C" {
uint32_t kmer_to_id(const char *kmer, int len) {
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

int fasta_count_kmers(int n_files, const char *fnames[], int K, int *total_kmer_counts) {
    int seqid = 0;
    uint64_t pos = 0;

    const uint64_t max_size = std::pow(4, K);
    ThreadPoolLocalData<int> pool;
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

            pool.queue_job([K, sequence, total_kmer_counts](int dum) { count_kmers(K, sequence, total_kmer_counts); });

            seqid++;
        }
    }
    pool.finalize();

    return 0;
}

int fasta_to_kmers_csr_cat_subseq(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                                  uint64_t *col_ind, uint64_t max_size, uint64_t *nnz, int *n_cols) {
    Timer timer;
    ThreadPoolLocalData<list_of_coo> pool;
    for (int seqid = 0; seqid < n_files; ++seqid) {
        pool.queue_job([K, seqid, fnames, L](list_of_coo &coo) {
            std::fstream stream;
            stream.open(fnames[seqid], std::ios::in);
            if (!stream)
                return;
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
                return;

            fill_indices_coo(coo, K, seqid, sequence);
        });
    }
    pool.finalize();
    timer.stop();
    std::cout << "Initial processing took: " << timer.elapsed() << " seconds\n";
    const uint64_t M = std::pow(4, K);
    pool.to_csr(M, row_ind, col_ind, data);
    *nnz = row_ind[M];
    *n_cols = n_files;
    return 0;
}

int fasta_to_kmers_csr(int n_files, const char *fnames[], int K, int L, float *data, uint64_t *row_ind,
                       uint64_t *col_ind, uint64_t max_size, uint64_t *nnz, int *n_cols) {
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

            pool.queue_job([K, seqid, sequence](list_of_coo &coo) { fill_indices_coo(coo, K, seqid, sequence); });

            seqid++;
        }
    }
    pool.finalize();

    pos = 0;
    uint64_t M = std::pow(4, K);

    pool.to_csr(M, row_ind, col_ind, data);

    *nnz = row_ind[M];
    *n_cols = seqid;
    return 0;
}


}
