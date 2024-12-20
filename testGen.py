import hashlib

salts = ["agr", "beg", "chr", "drg"]

def utf8_to_hex(input_str):
    # 將 UTF-8 字串轉換為十六進位字串
    return ''.join(f'{ord(char):02X}' for char in input_str)

def hex_to_ascii(hex_str):
    # 將每 4 個字節轉換為對應的 ASCII 字元
    if len(hex_str) % 4 != 0:
        raise ValueError("Hex string length must be a multiple of 4.")
    ascii_str = ''.join(chr(int(hex_str[i:i+4], 16)) for i in range(0, len(hex_str), 4))
    return ascii_str

def sha256_hash(input_data):
    # SHA-256 雜湊運算
    sha = hashlib.sha256()
    sha.update(input_data)
    return sha.digest()

def main():
    hashes_filename = "doc/hashes.txt"
    wordlist_filename = "doc/wordlist.txt"
    rules_filename = "doc/rules.txt"

    with open(wordlist_filename, "r") as wordlist_file:
        words = [line.strip() for line in wordlist_file]
    
    with open(rules_filename, "r") as rules_file:
        rules = [line.strip() for line in rules_file]

    for i in range(3):
        salt = salts[i % 4]
        complete_string = words[i] + salt
        hex_str = utf8_to_hex(complete_string)
        for _ in range(100): 
            hex_str = sha256_hash(hex_str.encode()).hex()
        show = hex_to_ascii(hex_str)
        print(f"{show}:{salt}")

if __name__ == "__main__":
    main()
