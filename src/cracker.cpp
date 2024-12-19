#include <hip/hip_runtime.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "util.h"
#include "rules.h"

__global__ void cracker_kernel(char* words, int words_num, char* hashes, int hashes_num,
                               char* rules, int rules_num, int* word_lengths_pre,
                               int* hash_lengths) {
    int block_idx = static_cast<dim3>(blockIdx).x;
    int thread_idx = static_cast<dim3>(threadIdx).x;
    int block_dim = static_cast<dim3>(blockDim).x;
    int grid_dim = static_cast<dim3>(gridDim).x;

    int idx = block_idx * block_dim + thread_idx;
    if (idx >= words_num) return;

    char* word = new char[word_lengths_pre[idx + 1] - word_lengths_pre[idx]];
    memcpy(word, words + word_lengths_pre[idx], word_lengths_pre[idx + 1] - word_lengths_pre[idx]);

    char* rule = new char[100];

    for (int rule_idx = 0; rule_idx < rules_num; rule_idx++) {
        memcpy(rule, rules + rule_idx, 100);
        char* candidate = new char[100];
        rules_apply_gpu(word, rule, candidate, word_lengths_pre[idx + 1] - word_lengths_pre[idx]);
    }
}

void launch_cracker(std::string hashes_filename, std::string& wordlist_filename,
                    std::vector<std::string>& rules) {
    std::ifstream hashes_file(hashes_filename);
    std::ifstream wordlist_file(wordlist_filename);

    std::vector<std::string> hashes;
    std::vector<int> hash_lengths;
    std::string hash;
    while (std::getline(hashes_file, hash)) {
        hashes.push_back(hash);
        hash_lengths.push_back(hash.size());
    }

    std::vector<std::string> words;
    std::vector<int> word_lengths;
    std::vector<int> word_lengths_pre;
    word_lengths_pre.push_back(0);
    std::string word;
    while (std::getline(wordlist_file, word)) {
        words.push_back(word);
        word_lengths.push_back(word.size());
        word_lengths_pre.push_back(word_lengths_pre.back() + word.size());
    }

    int rules_total_len = std::accumulate(rules.begin(), rules.end(), 0,
                                          [](int acc, std::string& rule) { return acc + rule.size(); });

    char* words_ptr = string_vector_to_char_array(words);
    char* hashes_ptr = string_vector_to_char_array(hashes);
    char* rules_ptr = string_vector_to_char_array(rules);

    int words_total_len = word_lengths_pre.back();
    int hashes_total_len = std::reduce(hash_lengths.begin(), hash_lengths.end());

    char* d_words;
    char* d_hashes;
    char* d_rules;
    int* d_word_lengths_pre;
    int* d_hash_lengths;

    hipMalloc(&d_words, words_total_len * sizeof(char));
    hipMalloc(&d_hashes, hashes_total_len * sizeof(char));
    hipMalloc(&d_rules, rules_total_len * sizeof(char));
    hipMalloc(&d_word_lengths_pre, word_lengths_pre.size() * sizeof(int));
    hipMalloc(&d_hash_lengths, hash_lengths.size() * sizeof(int));

    hipMemcpy(d_words, words_ptr, words_total_len * sizeof(char), hipMemcpyHostToDevice);
    hipMemcpy(d_hashes, hashes_ptr, hashes_total_len * sizeof(char), hipMemcpyHostToDevice);
    hipMemcpy(d_rules, rules_ptr, rules_total_len * sizeof(char), hipMemcpyHostToDevice);
    hipMemcpy(d_word_lengths_pre, word_lengths_pre.data(), word_lengths_pre.size() * sizeof(int),
              hipMemcpyHostToDevice);
    hipMemcpy(d_hash_lengths, hash_lengths.data(), hash_lengths.size() * sizeof(int),
              hipMemcpyHostToDevice);
}
