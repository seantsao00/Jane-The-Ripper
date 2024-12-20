#define RULE_LEN 100

#ifndef RULES_H
#define RULES_H

#include <hip/hip_runtime.h>

__host__ __device__ void rules_apply(const char* word, const char* rule, char* candidate, int word_length);

#endif
