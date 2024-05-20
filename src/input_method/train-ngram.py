# Train n-gram language model

import re

# Input Vocab: one char or two chars
alphabet = "abcdefghijklmnopqrstuvwxyz"
input_vocab = set(list(alphabet))
for i in range(26):
    for j in range(26):
        input_vocab.add(alphabet[i] + alphabet[j])
input_vocab = sorted(list(input_vocab))
print("Input Vocab:", input_vocab)

# load shakespere.txt
with open("shakespeare.txt") as f:
    text = f.read()
# text = text.lower()
text = re.sub(r'[^a-zA-Z]', ' ', text)
text = re.sub(r'\s+', ' ', text)
text = text[:10000]
print(text[:100])

output_vocab = text.split()
output_vocab = sorted(list(output_vocab))
print("Output Vocab:", output_vocab)

input_output_pairs = []
for word in output_vocab:
    if len(word) == 1:
        input_output_pairs.append((word, word))
    else:
        input_output_pairs.append((word[:2], word))

print("Number of pairs:", len(input_output_pairs))
print(input_output_pairs[:10])

# counter set
from collections import Counter
input_output_counter = Counter(input_output_pairs)
print("Number of unique pairs:", len(input_output_counter))
print(input_output_counter.most_common(10))

# convert_dict[input_char] = [(output_word, count), ...]
convert_dict = {}
for input_char, output_word in input_output_counter:
    if input_char not in convert_dict:
        convert_dict[input_char] = [(output_word, input_output_counter[(input_char, output_word)])]
    else:
        convert_dict[input_char].append((output_word, input_output_counter[(input_char, output_word)]))

def convert(input_chars_list, convert_dict):
    output_words = []
    print(input_chars_list)
    for char in input_chars_list:
        if char in convert_dict:
            print(char, convert_dict[char])
            output_word = max(convert_dict[char], key=lambda x: x[1])[0]
            output_words.append(output_word)
        else:
            output_words.append("UNK")

    output_sentence = " ".join(output_words)
    return output_sentence

input_sentence = "A twchinme"

input_words = input_sentence.split()
print(input_words)

# split the input words into two characters
input_chars_list = []
for word in input_words:
    i = 0
    while i < len(word):
        input_chars_list.append(word[i:i+2])
        i += 2
print(input_chars_list)

print(convert(input_chars_list, convert_dict))

print(convert_dict)