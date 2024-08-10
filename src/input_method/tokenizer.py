import re


# SimpleTokenizer for input
class TwoCharTokenizer:
    def __init__(self):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        vocab = {c + " ": i for i, c in enumerate(alphabet)}
        for i in range(len(alphabet)):
            for j in range(len(alphabet)):
                vocab[alphabet[i] + alphabet[j]] = len(vocab)
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str):
        text = input_encode(text)
        enc = []
        for i in range(0, len(text), 2):
            enc.append(text[i : i + 2])
        ids = [self.str_to_int[s] for s in enc]
        return ids

    def decode(self, ids: list):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


def input_encode(text):
    # input: "A two characters input method"
    # output: "a twchinme"
    text = text.lower()
    output = ""
    for word in text.split():
        if len(word) == 1:
            output += word + " "
        else:
            output += word[:2]
    return output


class WordTokenizer:
    def __init__(self, data_name="shakespeare"):
        vocab_size = 50304
        # freq = {word: count}
        with open(f"data/{data_name}/input.txt") as f:
            text = f.read()
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        from collections import Counter

        freq = Counter(text.split())
        vocab = {}
        for i, (word, _) in enumerate(freq.most_common(vocab_size)):
            vocab[word] = i
        # <unk> token
        vocab["<unk>"] = len(vocab)
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.vocab_size = len(vocab)

    def encode(self, text: str):
        text = text.lower()
        ids = [
            self.str_to_int[s] if s in self.str_to_int else self.str_to_int["<unk>"]
            for s in text.split()
        ]
        return ids

    def decode(self, ids: list):
        text = " ".join([self.int_to_str[i] for i in ids])
        return text


if __name__ == "__main__":
    vocab_size = 26 + 26 * 26

    text = "A two characters input method"

    print("input:", text)
    tokenizer = TwoCharTokenizer()
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)

    with open("data/shakespeare/input.txt") as f:
        text = f.read()

    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text[:100]
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)

    word_tokenizer = WordTokenizer(data_name="shakespeare")
    encoded = word_tokenizer.encode(text)
    decoded = word_tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
    print("vocab size", word_tokenizer.vocab_size)
