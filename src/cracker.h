#ifndef CRACKER_H
#define CRACKER_H

#include <hip/hip_runtime.h>

#define THREADS_PER_BLOCK 512
#define BLOCKS_PER_GRID 4096
#define ITERATIONS 1

__global__ void cracker_kernel(char* words, int words_idx, char* hash, char* rules, int rules_num,
                               int* word_lengths_pre, char* answer, char* salt);

void launch_cracker(std::string& hashes_filename, std::string& wordlist_filename,
                    std::string& rules_filename);
#endif