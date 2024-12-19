#include "rules.h"

#include <hip/hip_runtime.h>

#include <cctype>
#include <cstring>

__constant__ int max_candidate_len = 100;

bool strprefix(const char* str, const char* prefix) {
    return strncmp(prefix, str, strlen(prefix)) == 0;
}

__host__ __device__ void rules_apply(const char* word, const char* rule, char* candidate, int word_len) {
    strncpy(candidate, word, max_candidate_len - 1);
    candidate[max_candidate_len - 1] = '\0';
    int candidate_len = word_len;

    if (strcmp(rule, "l") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (isupper(candidate[i])) candidate[i] = tolower(candidate[i]);
        }
    } else if (strcmp(rule, "u") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (islower(candidate[i])) candidate[i] = toupper(candidate[i]);
        }
    } else if (strcmp(rule, "c") == 0) {
        if (islower(candidate[0])) candidate[0] = toupper(candidate[0]);
        for (int i = 1; i < candidate_len; i++) {
            if (isupper(candidate[i])) candidate[i] = tolower(candidate[i]);
        }
    } else if (strcmp(rule, "C") == 0) {
        if (isupper(candidate[0])) candidate[0] = tolower(candidate[0]);
        for (int i = 1; i < candidate_len; i++) {
            if (islower(candidate[i])) candidate[i] = toupper(candidate[i]);
        }
    } else if (strcmp(rule, "t") == 0) {
        for (int i = 0; i < candidate_len; i++) {
            if (islower(candidate[i])) {
                candidate[i] = toupper(candidate[i]);
            } else if (isupper(candidate[i])) {
                candidate[i] = tolower(candidate[i]);
            }
        }
    } else if (strprefix(rule, "T")) {
        int pos = strtol(rule + 2, nullptr, 10);
        if (pos < candidate_len) {
            if (islower(candidate[pos])) {
                candidate[pos] = toupper(candidate[pos]);
            } else if (isupper(candidate[pos])) {
                candidate[pos] = tolower(candidate[pos]);
            }
        }
    } else if (strcmp(rule, "r")) {
        for (int i = 0; i < candidate_len / 2; i++) {
            char tmp = candidate[i];
            candidate[i] = candidate[candidate_len - i - 1];
            candidate[candidate_len - i - 1] = tmp;
        }
    } else if (strcmp(rule, "d")) {
        strncpy(candidate + candidate_len, candidate, max_candidate_len - candidate_len);
        candidate_len = std::min(max_candidate_len, candidate_len * 2);
    } else if (strcmp(rule, "f")) {
        for (int i = 0; i < candidate_len && candidate_len + i < max_candidate_len; i++)
            candidate[candidate_len + i + 1] = candidate[i];
        candidate_len = std::min(max_candidate_len, candidate_len * 2);
    } else if (strcmp(rule, "{")) {
        char left_most = candidate[0];
        for (int i = 0; i < candidate_len - 1; i++) candidate[i] = candidate[i + 1];
        candidate[candidate_len - 1] = left_most;
    } else if (strcmp(rule, "}")) {
        char right_most = candidate[candidate_len - 1];
        for (int i = candidate_len - 1; i > 0; i--) candidate[i] = candidate[i - 1];
        candidate[0] = right_most;
    } else if (strcmp(rule, "p")) {
        if (candidate_len > 1) {
            int pos = candidate_len - 1;
            if (strchr("sxz", candidate[pos])
                || (pos > 1 && candidate[pos] == 'h'
                    && (candidate[pos - 1] == 'c' || candidate[pos - 1] == 's'))) {
                strcat(candidate, "es");
            } else if (candidate[pos] == 'f' && candidate[pos - 1] != 'f') {
                strcpy(&candidate[pos], "ves");
            } else if (pos > 1 && candidate[pos] == 'e' && candidate[pos - 1] == 'f') {
                strcpy(&candidate[pos - 1], "ves");
            } else if (pos > 1 && candidate[pos] == 'y') {
                if (strchr("aeiou", candidate[pos - 1])) {
                    strcat(candidate, "s");
                } else {
                    strcpy(&candidate[pos], "ies");
                }
            } else {
                strcat(candidate, "s");
            }
            candidate_len = strlen(candidate);
        }
    } else if (strcmp(rule, "P")) {
        int pos = candidate_len - 1;
        if (pos > 1 && !(candidate[pos] == 'd' && candidate[pos - 1] == 'e')) {
            if (candidate[pos] == 'y') {
                candidate[pos] = 'i';
            } else if (strchr("bgp", candidate[pos]) && !strchr("bgp", candidate[pos - 1])) {
                candidate[pos + 1] = candidate[pos];
                candidate[pos + 2] = 0;
            }

            if (candidate[pos] == 'e') {
                strcat(candidate, "d");
            } else {
                strcat(candidate, "ed");
            }
            candidate_len = strlen(candidate);
        }
    } else if (strcmp(rule, "I")) {
        int pos = candidate_len - 1;
        if (pos > 1
            && !(candidate[pos] == 'g' && candidate[pos - 1] == 'n' && candidate[pos - 2] == 'i')) {
            if (strchr("aeiou", candidate[pos])) {
                strcat(candidate, "ing");
            } else {
                if (strchr("bgp", candidate[pos]) && !strchr("bgp", candidate[pos - 1])) {
                    candidate[pos + 1] = candidate[pos];
                    candidate[pos + 2] = 0;
                }
                strcat(candidate, "ing");
            }
            candidate_len = strlen(candidate);
        }
    } else if (strprefix(rule, "A")) {
        const char* pos_start = rule + 2;
        char* end_pos;
        int n = strtol(pos_start, &end_pos, 10);
        const char* str = end_pos + 1;

        if (n >= 0 && n <= candidate_len) {
            int str_len = strlen(str);

            for (int i = candidate_len - 1; i >= n && candidate_len + str_len < max_candidate_len;
                 i--) {
                candidate[i + str_len] = candidate[i];
            }

            for (int i = 0; i < str_len && candidate_len + i < max_candidate_len; i++) {
                candidate[n + i] = str[i];
            }

            candidate_len = std::min(max_candidate_len, candidate_len + str_len);
            candidate[candidate_len] = '\0';

            candidate_len = strlen(candidate);
        }
    }
}