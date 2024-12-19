#include <hip/hip_runtime.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include "util.h"
#include "rules.h"
#include "sha256.h"

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 256
#define ITERATIONS 100

__global__ void cracker_kernel(char* words, int words_idx, char* hash, 
                               char* rules, int rules_num, int* word_lengths_pre,
                               char* answer, char* salt) {
    int block_idx = static_cast<dim3>(blockIdx).x;
    int thread_idx = static_cast<dim3>(threadIdx).x;
    int block_dim = static_cast<dim3>(blockDim).x;
    int grid_dim = static_cast<dim3>(gridDim).x;

    char* word = new char[word_lengths_pre[words_idx + block_idx + 1] - word_lengths_pre[words_idx + block_idx]];
    memcpy(word, words + word_lengths_pre[words_idx + block_idx], 
        word_lengths_pre[words_idx + block_idx + 1] - word_lengths_pre[words_idx + block_idx]);

    char* rule = new char[100];

    for (int rule_idx = thread_idx; rule_idx < (1 << rules_num); rule_idx += block_dim) {
        char* candidate = new char[100];
        candidate = word;
        for (int = 0; i < RULE_NUM; i++){
            if((i << i) & rule_idx){
                memcpy(rule, rules + i * 100, 100);
                rules_apply_gpu(word, rule, candidate, word_lengths_pre[block_idx + 1] - word_lengths_pre[block_idx]);
            }
        }
        candidate += salt;
        SHA256 ctx;
        char* tmp = new char[100];
        strcpy(tmp, candidate);
        for(int i = 0; i < ITERATIONS; i++){
            sha256(&ctx, reinterpret_cast<BYTE*>(tmp), strlen(tmp));
            tmp = reinterpret_cast<char*>(ctx.b);
        }
        if(!strcmp(reinterpret_cast<char*>(ctx.b), hash)){
            memcpy(answer, candidate, strlen(candidate));
        }
        delete[] tmp;
        delete[] candidate;
    }
    delete[] word;
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
    char* rules_ptr = string_vector_to_char_array(rules);
    char answer_ptr[100] = {};

    int words_total_len = word_lengths_pre.back();
    int words_num = word_lengths_pre.size() - 1;

    char* d_words;
    char* d_hash;
    char* d_rules;
    char* d_answer;
    int* d_word_lengths_pre;
    int* d_hash_lengths;

    hipMalloc(&d_words, words_total_len * sizeof(char));
    hipMalloc(&d_hash, 100 * sizeof(char));
    hipMalloc(&d_rules, rules_total_len * sizeof(char));
    hipMalloc(&d_word_lengths_pre, word_lengths_pre.size() * sizeof(int));
    hipMalloc(&d_hash_lengths, hash_lengths.size() * sizeof(int));
    hipMalloc(&d_answer, 100 * sizeof(char));

    for(hash: hashes){
        hipMemcpy(d_words, words_ptr, words_total_len * sizeof(char), hipMemcpyHostToDevice);
        hipMemcpy(d_hash, hash.data(), hash.size() * sizeof(char), hipMemcpyHostToDevice);
        hipMemcpy(d_rules, rules_ptr, rules_total_len * sizeof(char), hipMemcpyHostToDevice);
        hipMemcpy(d_word_lengths_pre, word_lengths_pre.data(), word_lengths_pre.size() * sizeof(int),
                hipMemcpyHostToDevice);
        hipMemcpy(d_hash_lengths, hash_lengths.data(), hash_lengths.size() * sizeof(int),
                hipMemcpyHostToDevice);
        hipMemcpy(d_answer, answer_ptr, 100 * sizeof(char), hipMemcpyHostToDevice);

        int words_idx = 0;
        while(words_idx < words_num){
            hipLaunchKernelGGL(cracker_kernel, dim3(BLOCKS_PER_GRID), dim3(THREADS_PER_BLOCK), 0, 0,
                            d_words, words_idx, d_hash, d_rules, RULE_NUM,
                            d_word_lengths_pre, d_answer, "salt");
            hipMemcpy(answer_ptr, d_answer, 100 * sizeof(char), hipMemcpyDeviceToHost);
            if (answer_ptr[0] != '\0') {
                std::cout << "Found password: " << answer_ptr << std::endl;
                break;
            }
            words_idx += BLOCKS_PER_GRID;
        }
    }
}
