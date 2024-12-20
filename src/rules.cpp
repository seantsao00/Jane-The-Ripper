#include "rules.h"

#include <hip/hip_runtime.h>

#include <cctype>
#include <cstring>

#include "util.h"

__host__ __device__ int max_candidate_len = 100;

__host__ __device__ void rules_apply(const char* word, const char* rule, char* candidate,
                                         int word_len) {
    memcpy(candidate, word, word_len + 1);
    candidate[max_candidate_len - 1] = '\0';
    int candidate_len = word_len;

    if (my_strcmp(rule, "l") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (my_isupper(candidate[i])) candidate[i] = my_tolower(candidate[i]);
        }
    } else if (my_strcmp(rule, "u") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (my_islower(candidate[i])) candidate[i] = my_toupper(candidate[i]);
        }
    } else if (my_strcmp(rule, "c") == 0) {
        if (my_islower(candidate[0])) candidate[0] = my_toupper(candidate[0]);
        for (int i = 1; i < candidate_len; i++) {
            if (my_isupper(candidate[i])) candidate[i] = my_tolower(candidate[i]);
        }
    } else if (my_strcmp(rule, "C") == 0) {
        if (my_isupper(candidate[0])) candidate[0] = my_tolower(candidate[0]);
        for (int i = 1; i < candidate_len; i++) {
            if (my_islower(candidate[i])) candidate[i] = my_toupper(candidate[i]);
        }
    } else if (my_strcmp(rule, "t") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (my_islower(candidate[i])) {
                candidate[i] = my_toupper(candidate[i]);
            } else if (my_isupper(candidate[i])) {
                candidate[i] = my_tolower(candidate[i]);
            }
        }
    } else if (strprefix(rule, "T")) {
        int pos = my_strtol(rule + 2, nullptr, 10);
        if (pos < candidate_len) {
            if (my_islower(candidate[pos])) {
                candidate[pos] = my_toupper(candidate[pos]);
            } else if (my_isupper(candidate[pos])) {
                candidate[pos] = my_tolower(candidate[pos]);
            }
        }
    } else if (my_strcmp(rule, "r") == 0) {
        for (int i = 0; i < candidate_len / 2; i++) {
            char tmp = candidate[i];
            candidate[i] = candidate[candidate_len - i - 1];
            candidate[candidate_len - i - 1] = tmp;
        }
    } else if (my_strcmp(rule, "d") == 0) {
        my_strncpy(candidate + candidate_len, candidate, max_candidate_len - candidate_len);
        candidate_len = std::min(max_candidate_len, candidate_len * 2);
    } else if (my_strcmp(rule, "f") == 0) {
        for (int i = 0; i < candidate_len && candidate_len + i < max_candidate_len; i++)
            candidate[candidate_len + i + 1] = candidate[i];
        candidate_len = std::min(max_candidate_len, candidate_len * 2);
    } else if (my_strcmp(rule, "{") == 0) {
        char left_most = candidate[0];
        for (int i = 0; i < candidate_len - 1; i++) candidate[i] = candidate[i + 1];
        candidate[candidate_len - 1] = left_most;
    } else if (my_strcmp(rule, "}") == 0) {
        char right_most = candidate[candidate_len - 1];
        for (int i = candidate_len - 1; i > 0; i--) candidate[i] = candidate[i - 1];
        candidate[0] = right_most;
    } else if (my_strcmp(rule, "p") == 0) {
        printf("candidate: %s\n", candidate);
        if (candidate_len > 1) {
            int pos = candidate_len - 1;
            if (my_strchr("sxz", candidate[pos])
                || (pos > 1 && candidate[pos] == 'h'
                    && (candidate[pos - 1] == 'c' || candidate[pos - 1] == 's'))) {
                printf("candidate: %s\n", candidate);
                printf("candidate_len: %d\n", candidate_len);
                my_strcat(candidate, "es");
            } else if (candidate[pos] == 'f' && candidate[pos - 1] != 'f') {
                my_strcpy(&candidate[pos], "ves");
            } else if (pos > 1 && candidate[pos] == 'e' && candidate[pos - 1] == 'f') {
                my_strcpy(&candidate[pos - 1], "ves");
            } else if (pos > 1 && candidate[pos] == 'y') {
                if (my_strchr("aeiou", candidate[pos - 1])) {
                    my_strcat(candidate, "s");
                } else {
                    my_strcpy(&candidate[pos], "ies");
                }
            } else {
                printf("candidate: %s\n", candidate);
                my_strcat(candidate, "s");
                printf("candidate: %s\n", candidate);
            }
            candidate_len = my_strlen(candidate);
        }
    } else if (my_strcmp(rule, "P") == 0) {
        int pos = candidate_len - 1;
        if (pos > 1 && !(candidate[pos] == 'd' && candidate[pos - 1] == 'e')) {
            if (candidate[pos] == 'y') {
                candidate[pos] = 'i';
            } else if (my_strchr("bgp", candidate[pos]) && !my_strchr("bgp", candidate[pos - 1])) {
                candidate[pos + 1] = candidate[pos];
                candidate[pos + 2] = 0;
            }

            if (candidate[pos] == 'e') {
                my_strcat(candidate, "d");
            } else {
                my_strcat(candidate, "ed");
            }
            candidate_len = my_strlen(candidate);
        }
    } else if (my_strcmp(rule, "I") == 0) {
        int pos = candidate_len - 1;
        if (pos > 1
            && !(candidate[pos] == 'g' && candidate[pos - 1] == 'n' && candidate[pos - 2] == 'i')) {
            if (my_strchr("aeiou", candidate[pos])) {
                my_strcat(candidate, "ing");
            } else {
                if (my_strchr("bgp", candidate[pos]) && !my_strchr("bgp", candidate[pos - 1])) {
                    candidate[pos + 1] = candidate[pos];
                    candidate[pos + 2] = 0;
                }
                my_strcat(candidate, "ing");
            }
            candidate_len = my_strlen(candidate);
        }
    } else if (strprefix(rule, "A") == 0) {
        const char* pos_start = rule + 2;
        char* end_pos;
        int n = my_strtol(pos_start, &end_pos, 10);
        const char* str = end_pos + 1;

        if (n >= 0 && n <= candidate_len) {
            int str_len = my_strlen(str);

            for (int i = candidate_len - 1; i >= n && candidate_len + str_len < max_candidate_len;
                 i--) {
                candidate[i + str_len] = candidate[i];
            }

            for (int i = 0; i < str_len && candidate_len + i < max_candidate_len; i++) {
                candidate[n + i] = str[i];
            }

            candidate_len = std::min(max_candidate_len, candidate_len + str_len);
            candidate[candidate_len] = '\0';

            candidate_len = my_strlen(candidate);
        }
    }
}