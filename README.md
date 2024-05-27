# input-method
Two character input method.

## How to use
- train tokenizer
```bash
python3 src/input_method/train_tokenizer.py --data_name "shakespeare"
```


## LLM-based Two Characters Input Method

- Strip only the first two characters from the input string
- Tokenize the input string
- Predict the whole string using the LLM


inputのtokenと, outputのtokenが異なる.

Input: A twchinme

Stripped input: A tw ch in me

TwoCharTokezer encode: 203 40 23 10 3

Model output: 100 23 40 19 3
BPETokenizer Decode: A two characters input method

### Create Dataset
Text: A two characters input method
Input: A tw ch in me
Input tokenized: TwoCharTokenizer("A tw ch in me")
Output: A two characters input method
Output tokenized: BPETokenizer("A two characters input method")

Training Dataset:
(SimpleTokenizer("A tw ch in me"), BPETokenizer("A two characters input method")), ...


## How to use


### N-gram model

- train and evaluate the n-gram model with shakespeare dataset
```bash
python3 src/input_method/train-ngram.py 
```

### Transformer based language modelf
- train the NanoLM model with shakespeare dataset
```bash
python3 src/input_method/train.py --data_name "shakespeare" --batch_size 128 --n_iteration
s 5000 --n_freq_eval 100 --dropout_rate 0.1 --learning_rate 0.001 --num_layers 8 --embed_size 256  --head_size 32 --num_heads 8 --block_
size 1
```



## Point
- Two tokenizer is needed
  - One for input: This tokenizer is simple one(one character and two character tokenization. the number of tokens is 26+26*26=702)
  - One for output: We use BPE tokenizer for output tokenization. The number of tokens is 50304.
- P(characters | a tw ch)
