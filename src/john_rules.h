#ifndef JOHN_RULES_H
#define JOHN_RULES_H

#include <hip/hip_runtime.h>

__device__ void rules_apply(const char* word, const char* rule, char* candidate,
                                int word_length);

#endif