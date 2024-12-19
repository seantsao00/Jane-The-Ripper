#include <iostream>
#include "rules.h"
#include "sha256.h"
#include "util.h"
#include "cracker.h"

int main() {
    std::string hashes_filename = "../doc/hashes.txt";
    std::string wordlist_filename = "../doc/wordlist.txt";
    std::string rules_filename = "../doc/rules.txt";
    launch_cracker(hashes_filename, wordlist_filename, rules_filename);
    return 0;
}
