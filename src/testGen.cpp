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

    std::vector<std::string> words;
    std::vector<std::string> rules;
    std::string word;
    std::string rule;

    while (std::getline(wordlist_file, word)) words.push_back(word); 
    while (std::getline(rules_file, rule)) rules.push_back(rule);
    
    for (int i = 0; i < 1; i++) {
        std::string salt = salts[i%4];
        std::string complete_string = words[i] + salt;
        char candidate[100];
        char hex[100];
        printf("%d\n", complete_string.size());
        rules_apply(complete_string.data(), rules[i%4].c_str(), candidate,
                        complete_string.size());
        printf("candidate: %s\n", candidate);
        utf8ToHex(candidate, hex);
        printf("hex: %s\n", hex);
        SHA256 ctx;
        sha256(&ctx, (BYTE*)(hex), strlen(hex));
        memcpy(hex, ctx.b, 32);
        char show[100];
        hexToAscii(hex, show);
        std::cout << show << ":" << salt << '\n';
    }
}