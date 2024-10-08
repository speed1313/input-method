# input-method
First-two-char input method using transformer-based language model and n-gram model.
The model predicts the corresponding word ("method") given the previous and current two characters ("fi", "tw", "ch", "in", "me").

<img src="./figure/two_char_lm_overview.png" width="50%">


## How to use

### Data preparation
- Download the shakespeare dataset
```bash
$ wget https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt -P data/shakespeare
```


### N-gram model

- train and evaluate the n-gram model with shakespeare dataset
```bash
$ python3 src/input_method/train-ngram.py --ngram 2 --input "My name is Taro. I am a student."
```

### Transformer-based language model
- Train the NanoLM model with shakespeare dataset
```bash
$ python3 src/input_method/train.py --data_name "shakespeare" --batch_size 128 --n_iterations 5000 --n_freq_eval 100 --dropout_rate 0.1 --learning_rate 0.001 --num_layers 8 --embed_size 256  --head_size 32 --num_heads 8 --block_size 4
```

- Evaluate the NanoLM model with shakespeare dataset
```bash
$ python3 src/input_method/evaluate.py --data_name "shakespeare" --block_size 4
```

- Sequence to sequence prediction
```bash
$ python3 src/input_method/seq_to_seq.py --data_name "shakespeare" --block_size 16 --input "My name is Taro. I am a student."

Prompt: My name is Taro. I am a student.
Output: my name is taken i am a strange
```
This program internally convert the prompt to the first-two-char input format and predict the corresponding word using the trained NanoLM model sequentially.

- Training on Wikitext2

```bash
python3 src/input_method/prepare_wikitext.py
```


```bash
$ python3 src/input_method/train.py --data_name "wikitext" --batch_size 1024 --n_iterations 1000 --n_freq_eval 100 --dropout_rate 0.0 --learning_rate 0.001 --num_layers 8 --embed_size 256  --head_size 32 --num_heads 8 --block_size 4
```


## Features
- Two tokenizers are used
  - TwoCharTokenizer: vocab = {"a ", ..., "z ", "aa", ..., "zz"}
      - The vocab size is 26 + 26 * 26 = 702
  - WordTokenizer: vocab = {"a", ..., "word", ...}
    - The vocab size depends on the dataset
- Predict the corresponding word given the previous and current two characters (e.g., P("method" | ("a ", "tw", "ch", "in", "me"))) using the transformer-based language model

## Results

- N-gram model
![n-gram](./figure/n-gram.png)

- Transformer-based language model
![transformer](./figure/transformer-based.png)

2-gram model was the best model among all models.

## Draft Paper
You can access the draft paper about this project [here](./draft_paper.pdf).

## Citation
```
@article{sugiura2024input,
  title   = "First-two-char Input Method with N-gram Model and
Transformer-based Language Model",
  author  = "Issa, Sugiura",
  journal = "github.com",
  year    = "2024",
  month   = "Aug",
  url     = "https://github.com/speed1313/input-method"
}
```




## Reference
- https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt
- https://github.com/speed1313/jax-llm
- Vaswani et al. "Attention is All You Need." NeurIPS 2017.
-
