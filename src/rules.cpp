#include "rules.h"

#include <hip/hip_runtime.h>

#include <cctype>
#include <cstring>

bool strprefix(const char* str, const char* prefix) {
    return strncmp(prefix, str, strlen(prefix)) == 0;
}

__device__ void rules_apply_gpu(const char* word, const char* rule, char* candidate, int word_len) {
    int len = 0;
    while (word[len] && len < word_len - 1) {
        candidate[len] = word[len];
        len++;
    }
    candidate[len] = '\0';

    if (strcmp(rule, "l") == 0) {
        for (int i = 0; i < len; i++) {
            if (isupper(candidate[i])) candidate[i] = tolower(candidate[i]);
        }
    } else if (strcmp(rule, "u") == 0) {
        for (int i = 0; i < len; i++) {
            if (islower(candidate[i])) candidate[i] = toupper(candidate[i]);
        }
    } else if (strcmp(rule, "c") == 0) {
        if (islower(candidate[0])) candidate[0] = toupper(candidate[0]);
        for (int i = 1; i < len; i++) {
            if (isupper(candidate[i])) candidate[i] = tolower(candidate[i]);
        }
    } else if (strcmp(rule, "C") == 0) {
        if (isupper(candidate[0])) candidate[0] = tolower(candidate[0]);
        for (int i = 1; i < len; i++) {
            if (islower(candidate[i])) candidate[i] = toupper(candidate[i]);
        }
    } else if (strcmp(rule, "t") == 0) {
        for (int i = 0; i < len; i++) {
            if (islower(candidate[i])) {
                candidate[i] = toupper(candidate[i]);
            } else if (isupper(candidate[i])) {
                candidate[i] = tolower(candidate[i]);
            }
        }
    } else if (strprefix(rule, "T")) {
        int pos
    }
}