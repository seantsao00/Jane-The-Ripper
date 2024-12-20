# Jane-The-Ripper

## Prerequisites

To run *Jane-The-Ripper*, ensure you have the following:
- An AMD GPU with ROCm support
- `hipcc` (ROCm's HIP compiler)
- A Linux environment with `make` installed

## Usage

### 1. Cracking Passwords

To crack password hashes provided in `doc/hashes.txt`:

1. **Compile the Cracker**  
   Run the following command to compile the main cracking application:
   ```bash
   make jane
   ```

2. **Start Cracking**  
   Execute the application to begin cracking the hashes:
   ```bash
   ./jane
   ```
   The results will be printed to the standard output.

---

### 2. Generating Hashes

To generate password hashes from a wordlist:

1. **Compile the Hash Generator**  
   Run the following command to compile the test hash generator:
   ```bash
   make testGen
   ```

2. **Generate Hashes**  
   Run the following command to generate hashed values from the wordlist:
   ```bash
   ./testGen
   ```
   The generated hash values will be saved in `doc/hashes.txt`.

---

### 3. Customizing the Wordlist

To customize the wordlist:
- Open `doc/wordlist.txt` in any text editor.
- Add, edit, or remove words to adjust the list of passwords to be tested.

---

### 4. Customizing Rules

To define or modify rules for password transformations:
- Edit the `doc/rules.txt` file.
- Each line represents a rule. Below is a detailed guide to the available rules:

#### **Available Rules**
| Rule   | Description                                                                 |
|--------|-----------------------------------------------------------------------------|
| `l`    | Convert all characters to lowercase.                                       |
| `u`    | Convert all characters to uppercase.                                       |
| `c`    | Capitalize the first character and lowercase the rest.                     |
| `C`    | Lowercase the first character and uppercase the rest.                      |
| `t`    | Toggle the case of all characters.                                         |
| `T n`  | Toggle the case of the character at position `n`.                          |
| `r`    | Reverse the word (e.g., `"Fred"` → `"derF"`).                              |
| `d`    | Duplicate the word (e.g., `"Fred"` → `"FredFred"`).                        |
| `f`    | Reflect the word (e.g., `"Fred"` → `"FredderF"`).                          |
| `{`    | Rotate the word left (e.g., `"jsmith"` → `"smithj"`).                      |
| `}`    | Rotate the word right (e.g., `"smithj"` → `"jsmith"`).                     |
| `p`    | Pluralize the word (e.g., `"crack"` → `"cracks"`).                         |
| `P`    | Convert to past tense (e.g., `"crack"` → `"cracked"`).                     |
| `I`    | Convert to present participle (e.g., `"crack"` → `"cracking"`).            |
| `A n str` | Add string `str` at position `n`. Supports subsets of regex:            |
|        | - `[a-z]` for alphabetic sets                                              |
|        | - `[ab]` for specific character sets                                       |
|        | - `[a-z]{1,3}` for repetitions (1 to 3 times).                             |

---

### Example Workflow

1. **Edit Your Wordlist**  
   Add potential passwords to `doc/wordlist.txt`.

2. **Define Your Rules**  
   Customize `doc/rules.txt` to reflect the transformations you want to apply.

3. **Generate Test Hashes**  
   Run:
   ```bash
   ./testGen
   ```

4. **Crack the Passwords**  
   Execute:
   ```bash
   ./jane
   ```
   View the cracked passwords in the standard output.

---

## Notes

- Ensure that the wordlist and rules are formatted correctly.
- Larger wordlists and more complex rules will increase runtime but improve the chances of cracking passwords.
- The tool is optimized for AMD GPUs with ROCm but should be extensible for other architectures with minor adjustments.

