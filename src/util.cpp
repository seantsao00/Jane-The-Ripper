#include "util.h"

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