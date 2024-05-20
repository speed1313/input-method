# Input: A twchinme
# Output: A two character input method

input_sentence = "A twchinme"
output_sentence = "A two character input method"

# split the input sentence into words
input_words = input_sentence.split()
print(input_words)

# split the input words into two characters
input_chars_list = []
for word in input_words:
    i = 0
    while i < len(word):
        input_chars_list.append(word[i : i + 2])
        i += 2
print(input_chars_list)

# convert two characters into a word
# TODO: We need to train this dictionary
output_words = []
convert_dict = {"A": "A", "tw": "two", "ch": "character", "in": "input", "me": "method"}

output_words = [convert_dict[char] for char in input_chars_list]
print(output_words)
output_sentence = " ".join(output_words)
print(output_sentence)
