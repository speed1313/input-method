# Train n-gram language model

import re
from tokenizer import TwoCharTokenizer, WordTokenizer

# Input Vocab: one char or two chars
text = "A two characters input method"

input_tokenizer = TwoCharTokenizer()
input_tokenized_text = input_tokenizer.encode(text)
print("text:", text)
print("encoded:", input_tokenized_text)
print("decoded:", input_tokenizer.decode(input_tokenized_text))

output_tokenizer = WordTokenizer()
output_tokenized_text = output_tokenizer.encode(text)
print("text:", text)
print("encoded:", output_tokenized_text)
print("decoded:", output_tokenizer.decode(output_tokenized_text))


# load shakespere.txt
with open("data/shakespeare/input.txt") as f:
    text = f.read()
# text = text.lower()
text = re.sub(r"[^a-zA-Z]", " ", text)
text = re.sub(r"\s+", " ", text)
print("fist part of the text:", text[:100])
input_tokenized_text = input_tokenizer.encode(text)
output_tokenized_text = output_tokenizer.encode(text)
train_ratio = 0.90
train_x = input_tokenized_text[: int(train_ratio * len(input_tokenized_text))]
train_y = output_tokenized_text[: int(train_ratio * len(output_tokenized_text))]
eval_x = input_tokenized_text[int(train_ratio * len(input_tokenized_text)) :]
eval_y = output_tokenized_text[int(train_ratio * len(output_tokenized_text)) :]


# ngram model
# p("Two" | "tw")
from collections import Counter, defaultdict


def predict_model(model, prefix):
    if prefix in model:
        suffix = model[prefix]
        return model[prefix].most_common(1)[0][0]
    else:
        for i in range(1, len(prefix)):
            if prefix[i:] in model:
                return model[prefix[i:]].most_common(1)[0][0]
        return None


# ngram model
# p("Two" | "tw")


def train_ngram_model(input_tokenized_text, output_tokenized_text, n=2):
    ngram_dict = defaultdict(Counter)
    # ngram_dict = {"tw": {"two": 1}, "wo": {"words": 1}}
    for i in range(n):
        for j in range(i, len(input_tokenized_text)):
            prefix = input_tokenized_text[j - i : j + 1]
            suffix = output_tokenized_text[j]
            prefix = tuple(prefix)
            ngram_dict[prefix][suffix] += 1
    return ngram_dict


N = 6

model = train_ngram_model(train_x, train_y, n=N)
# tupled


def evaluate_ngram_model(model, eval_x, eval_y, input_tokenizer, output_tokenizer, n=3):
    test_accuracy = 0
    unk_num = 0
    # n window
    for i in range(len(eval_x)):
        if i < n - 1:
            prefix = tuple(eval_x[: i + 1])
        else:
            prefix = tuple(eval_x[i - n + 1 : i + 1])
        predict = predict_model(model, prefix)
        if predict == eval_y[i]:
            test_accuracy += 1
        if predict is None:
            unk_num += 1
    test_accuracy /= len(eval_x)
    print("unk_num:", unk_num)
    return test_accuracy


def ngram_model(model, input_text: str = "A two characters input method", n=3) -> str:
    input_tokenized_text = input_tokenizer.encode(input_text)
    predict = []
    for i in range(len(input_tokenized_text)):
        if i < n - 1:
            prefix = tuple(input_tokenized_text[: i + 1])
        else:
            prefix = tuple(input_tokenized_text[i - n + 1 : i + 1])
        output = predict_model(model, prefix)
        predict.append(output)
    decoded = output_tokenizer.decode(predict)
    return decoded


# n>=3 で過学習している
for n in range(1, N + 1):
    print("n:", n)
    train_accuracy = evaluate_ngram_model(
        model, train_x, train_y, input_tokenizer, output_tokenizer, n
    )
    test_accuracy = evaluate_ngram_model(
        model, eval_x, eval_y, input_tokenizer, output_tokenizer, n
    )
    print("Train accuracy:", train_accuracy)
    print("Test accuracy:", test_accuracy)
    # predict = ngram_model(model, "A two characters input method", n)
    # print("Predict:", predict)
    print()
