#ifndef UTIL_H
#define UTIL_H

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

char* string_vector_to_char_array(std::vector<std::string>& strings);

__device__ __host__ bool strprefix(const char* str, const char* prefix);

__host__ __device__ char* my_strncpy(char* dest, const char* src, size_t n);

__host__ __device__ int my_strcmp(const char* str1, const char* str2);

__host__ __device__ bool my_isupper(char c);

__host__ __device__ bool my_islower(char c);

__host__ __device__ char my_tolower(char c);

__host__ __device__ char my_toupper(char c);

__host__ __device__ char* my_strchr(const char* str, int c);

__host__ __device__ long my_strtol(const char* str, char** endptr, int base);

__host__ __device__ char* my_strcat(char* dest, const char* src);

__host__ __device__ char* my_strcpy(char* dest, const char* src);

__host__ __device__ size_t my_strlen(const char* str);

void hexToAscii(const char *hex, char *ascii);

void utf8ToHex(const char *utf8Str, char *hexOutput);

#endif
