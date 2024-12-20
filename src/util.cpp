#include "util.h"

#include <hip/hip_runtime.h>

#include <cstring>
#include <numeric>
#include <string>
#include <vector>

char* string_vector_to_char_array(std::vector<std::string>& strings) {
    std::vector<int> lengths;
    for (int i = 0; i < strings.size(); i++) lengths.push_back(strings[i].size());

    char* strings_ptr = new char[std::reduce(lengths.begin(), lengths.end())];
    int offset = 0;
    for (int i = 0; i < strings.size(); i++) {
        memcpy(strings_ptr + offset, strings[i].c_str(), strings[i].size());
        offset += strings[i].size();
    }

    return strings_ptr;
}

__device__ __host__ bool strprefix(const char* str, const char* prefix) {
    while (*prefix) {
        if (*str++ != *prefix++) {
            return false;
        }
    }
    return true;
}

__host__ __device__ char* my_strncpy(char* dest, const char* src, size_t n) {
    size_t i = 0;

    // Copy characters from src to dest up to n or until null-terminator
    for (; i < n && src[i] != '\0'; ++i) {
        dest[i] = src[i];
    }

    // Null-terminate dest if length of src is less than n
    for (; i < n; ++i) {
        dest[i] = '\0';
    }

    return dest;
}

// Custom strcmp implementation
__host__ __device__ int my_strcmp(const char* str1, const char* str2) {
    while (*str1 && (*str1 == *str2)) {
        str1++;
        str2++;
    }
    return *(unsigned char*)str1 - *(unsigned char*)str2;
}

// Custom isupper implementation
__host__ __device__ bool my_isupper(char c) { return c >= 'A' && c <= 'Z'; }

// Custom tolower implementation
__host__ __device__ char my_tolower(char c) {
    if (my_isupper(c)) {
        return c + ('a' - 'A');
    }
    return c;
}

// Custom islower implementation
__host__ __device__ bool my_islower(char c) { return c >= 'a' && c <= 'z'; }

// Custom toupper implementation
__host__ __device__ char my_toupper(char c) {
    if (my_islower(c)) {
        return c - ('a' - 'A');
    }
    return c;
}

__host__ __device__ char* my_strchr(const char* str, int c) {
    while (*str) {
        if (*str == c) {
            return (char*)str;
        }
        str++;
    }
    return c == '\0' ? (char*)str : nullptr;
}

// Custom strtol implementation
__host__ __device__ long my_strtol(const char* str, char** endptr, int base) {
    long result = 0;
    int sign = 1;

    // Skip whitespace
    while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r' || *str == '\v'
           || *str == '\f') {
        str++;
    }

    // Check for sign
    if (*str == '-') {
        sign = -1;
        str++;
    } else if (*str == '+') {
        str++;
    }

    // Parse digits
    while (*str) {
        int digit;
        if (*str >= '0' && *str <= '9') {
            digit = *str - '0';
        } else if (*str >= 'a' && *str <= 'z') {
            digit = *str - 'a' + 10;
        } else if (*str >= 'A' && *str <= 'Z') {
            digit = *str - 'A' + 10;
        } else {
            break;
        }

        if (digit >= base) {
            break;
        }

        result = result * base + digit;
        str++;
    }

    if (endptr) {
        *endptr = (char*)str;
    }
    return result * sign;
}

// Custom strcat implementation
__host__ __device__ char* my_strcat(char* dest, const char* src) {
    char* ptr = dest;

    // Move to the end of dest
    while (*ptr) {
        ptr++;
    }

    // Copy src to the end of dest
    while (*src) {
        *ptr++ = *src++;
    }

    // Null-terminate dest
    *ptr = '\0';

    return dest;
}

// Custom strcpy implementation
__host__ __device__ char* my_strcpy(char* dest, const char* src) {
    char* ptr = dest;

    while (*src) {
        *ptr++ = *src++;
    }

    *ptr = '\0';

    return dest;
}

// Custom strlen implementation
__host__ __device__ size_t my_strlen(const char* str) {
    size_t len = 0;

    while (*str++) {
        len++;
    }

    return len;
}

void utf8ToHex(const char* utf8Str, char* hexOutput, const int len) {
    size_t hexIndex = 0;

    for (size_t i = 0; i < len; ++i) {
        unsigned char byte = (unsigned char)utf8Str[i];
        sprintf(&hexOutput[hexIndex], "%02X", byte);
        hexIndex += 2;
    }

    hexOutput[hexIndex] = '\0';
}

void hexToUtf8(const char* hexStr, char* utf8Output) {
    size_t len = strlen(hexStr);

    // Ensure the length of the hex string is even
    if (len % 2 != 0) {
        fprintf(stderr, "Error: Hex string length must be even.\n");
        utf8Output[0] = '\0';
        return;
    }

    size_t utf8Index = 0;

    for (size_t i = 0; i < len; i += 2) {
        char hexByte[3] = { hexStr[i], hexStr[i + 1], '\0' };

        // Validate hex characters
        if (!isxdigit(hexByte[0]) || !isxdigit(hexByte[1])) {
            fprintf(stderr, "Error: Invalid hex character found.\n");
            utf8Output[0] = '\0';
            return;
        }

        unsigned char byte = (unsigned char)strtol(hexByte, NULL, 16);
        utf8Output[utf8Index++] = (char)byte;
    }

    utf8Output[utf8Index] = '\0';
}
