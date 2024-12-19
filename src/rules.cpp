#include "rules.h"

#include <hip/hip_runtime.h>

#include <cstring>

#include "john_rules.h"

__device__ void rules_apply_gpu(const char* word, const char* rule, char* candidate,
                                int word_length) {
    // Copy the input word into the candidate buffer
    int length = 0;
    while (word[length] && length < word_length - 1) {
        candidate[length] = word[length];
        length++;
    }
    candidate[length] = '\0';  // Null-terminate the candidate

    // Process each command in the rule
    rules_apply(word, rule, candidate, word_length);
}