#include "sha256.h"
#include "rules.h"
#include "util.h"
#include "cracker.h"
#include <iostream>
#include <fstream>

std::string salts[] = {"agr", "beg", "chr", "drg"};

int main()
{
    std::string hashes_filename = "../doc/hashes.txt";
    std::string wordlist_filename = "../doc/wordlist.txt";
    std::string rules_filename = "../doc/rules.txt";

    std::ifstream wordlist_file(wordlist_filename);
    std::ifstream rules_file(rules_filename);

    std::vector<std::string> words;
    std::vector<std::string> rules;
    std::string word;
    std::string rule;
    
    while (std::getline(wordlist_file, word))
        words.push_back(word);
    while (std::getline(rules_file, rule))
        rules.push_back(rule);

    std::random_device rd;
    std::mt19937 gen1(rd());
    std::uniform_int_distribution<> dist1(0, words.size() - 1);

    std::mt19937 gen2(rd());
    std::uniform_int_distribution<> dist2(0, rules.size() - 1);

    std::mt19937 gen3(rd());
    std::uniform_int_distribution<> dist3(0, salts.size() - 1);

    for(int i=0; i<100; i++){
        std::string complete_string = words[dist1(gen1)] + salts[dist3(gen3)];
        rules_apply(complete_string.c_str(), rules[dist2(gen2)].c_str(), nullptr, complete_string.size());
        for()
    }
}