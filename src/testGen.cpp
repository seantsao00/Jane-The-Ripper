#include <fstream>
#include <iostream>
#include <random>

#include "cracker.h"
#include "rules.h"
#include "sha256.h"
#include "util.h"

std::array<std::string, 4> salts = {"agr", "beg", "chr", "drg"};

int main() {
    std::string hashes_filename = "../doc/hashes.txt";
    std::string wordlist_filename = "../doc/wordlist.txt";
    std::string rules_filename = "../doc/rules.txt";

    std::ifstream wordlist_file(wordlist_filename);
    std::ifstream rules_file(rules_filename);

    std::vector<std::string> words;
    std::vector<std::string> rules;
    std::string word;
    std::string rule;

    while (std::getline(wordlist_file, word)) words.push_back(word);
    while (std::getline(rules_file, rule)) rules.push_back(rule);

    std::random_device rd;
    std::mt19937 gen1(rd());
    std::uniform_int_distribution<> dist1(0, words.size() - 1);

    std::mt19937 gen2(rd());
    std::uniform_int_distribution<> dist2(0, rules.size() - 1);

    std::mt19937 gen3(rd());
    std::uniform_int_distribution<> dist3(0, salts.size() - 1);

    for (int i = 0; i < 100; i++) {
        std::string salt = salts[dist3(gen3)];
        std::string complete_string = words[dist1(gen1)] + salt;
        char candidate[100];
        rules_apply_gpu(complete_string.c_str(), rules[dist2(gen2)].c_str(), candidate,
                    complete_string.size());
        SHA256 ctx;
        for (int j = 0; j < ITERATIONS; j++) {
            sha256(&ctx, reinterpret_cast<BYTE*>(candidate), strlen(candidate));
            memcpy(candidate, ctx.b, 32);
        }
        std::cout << candidate << ":" << salt << std::endl;
    }
}