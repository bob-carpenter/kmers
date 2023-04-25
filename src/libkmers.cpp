#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

class ThreadPool {
  public:
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
    bool busy() {
        bool poolbusy;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            poolbusy = jobs.empty();
        }
        return poolbusy;
    }

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

void atomic_increment(int *arr, size_t index) {
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

void fill_indices(int K, int seq_total_count, int seqid, const std::string &sequence, float *data, int *row_ind,
                  int *col_ind, int *total_kmer_counts) {
    const auto kmer_counts = collect_kmers(sequence.data(), K);

    int pos = 0;
    for (auto &[kmer, count] : kmer_counts) {
        data[pos] = float(count) / seq_total_count;

        row_ind[pos] = kmer;
        col_ind[pos] = seqid;
        atomic_increment(total_kmer_counts, kmer);
        pos++;
    }
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

inline int get_total_count(int L, int K) { return L <= K ? 0 : L - K + 1; }

extern "C" {
int fasta_to_kmers_sparse(const char *fname, int K, float *data, int *row_ind, int *col_ind, int *total_kmer_counts,
                          int max_size, int *n_elements, int *n_cols) {
    std::fstream stream;
    stream.open(fname, std::ios::in);
    if (!stream)
        return -1;

    int seqid = 0;
    int pos = 0;
    ThreadPool pool;
    pool.Start();

    while (true) {
        std::string header, sequence;
        std::tie(header, sequence) = get_next_sequence(stream);
        if (!header.length())
            break;
        if (header.find("PREDICTED") != std::string::npos)
            continue;

        int seq_total_count = get_total_count(sequence.length(), K);
        pool.QueueJob([K, seq_total_count, seqid, sequence, data, row_ind, col_ind, total_kmer_counts, pos]() {
            fill_indices(K, seq_total_count, seqid, sequence, data + pos, row_ind + pos, col_ind + pos,
                         total_kmer_counts);
        });

        if (seqid % 10000 == 0)
            std::cout << seqid << " " << pos << std::endl;

        pos += seq_total_count;
        seqid++;
    }

    pool.Stop();
    *n_elements = pos;
    *n_cols = seqid;
    return 0;
}
}
