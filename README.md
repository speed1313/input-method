# input-method
Two character input method.


## LLM-based Two Characters Input Method

- Strip only the first two characters from the input string
- Tokenize the input string
- Predict the whole string using the LLM


inputのtokenと, outputのtokenが異なる.

Input: A twchinme

Stripped input: A tw ch in me

Tokenized input: 203 40 23 10 3

Model output: 100 23 40 19 3
Decoded output: A two characters input method


### Create Dataset
Text: A two characters input method
Input: A tw ch in me
Input tokenized: SimpleTokenizer("A tw ch in me")
Output: A two characters input method
Output tokenized: BPETokenizer("A two characters input method")

Training Dataset:
(SimpleTokenizer("A tw ch in me"), BPETokenizer("A two characters input method")), ...





## Point
- Two tokenizer is needed
  - One for input: This tokenizer is simple one(one character and two character tokenization. the number of tokens is 26+26*26=702)
  - One for output: We use BPE tokenizer for output tokenization. The number of tokens is 50304.
- P(characters | a tw ch)
