#define RULE_NUM 10

#ifndef RULES_H
#define RULES_H

#include <hip/hip_runtime.h>

__device__ void rules_apply_gpu(const char* word, const char* rule, char* candidate, int word_length);

#endif
