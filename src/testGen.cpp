#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <iomanip>

#include "cracker.h"
#include "rules.h"
#include "sha256.h"
#include "util.h"

std::array<std::string, 4> salts = {"agr", "beg", "chr", "drg"};

int main() {
    std::string hashes_filename = "doc/hashes.txt";
    std::string wordlist_filename = "doc/wordlist.txt";
    std::string rules_filename = "doc/rules.txt";

    std::ifstream wordlist_file(wordlist_filename);
    std::ifstream rules_file(rules_filename);
    std::ofstream hashes_file(hashes_filename);

    std::vector<std::string> words;
    std::vector<std::string> rules;
    std::string word;
    std::string rule;

    while (std::getline(wordlist_file, word)) words.push_back(word);
    while (std::getline(rules_file, rule)) rules.push_back(rule);
    
    for (int i = 0; i < 100; i++) {
        std::string salt = salts[i%4];
        std::string complete_string = words[i] + salt;
        complete_string.resize(100, '\0');
        // std::cout << complete_string << '\n';
        char candidate[100];
        char hex[100], tmp[100];
        rules_apply(complete_string.data(), rules[i%7].data(), candidate,
                        complete_string.size());
        SHA256 ctx;
        sha256(&ctx, (BYTE*)(candidate), strlen(candidate));
        
        memcpy(tmp, ctx.b, 32);
        tmp[32] = '\0';
        char show[100];
        utf8ToHex(tmp, show, 32);

        for (int i = 0; show[i]; i++)
            if(isalpha(show[i]))
                show[i] = tolower((unsigned char)show[i]);

        hashes_file << show << ":" << salt << '\n';
    }
}