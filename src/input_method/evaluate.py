import jax
import jax.numpy as jnp
from model import NanoLM
import pickle
import json
import click
from tokenizer import TwoCharTokenizer, WordTokenizer
import re
import os


@click.command()
@click.option("--data_name", type=str, default="shakespeare")
def main(
    data_name: str,
):
    model_path = f"model/{data_name}"
    # load config json
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    # load params
    with open(f"{model_path}/params.pkl", "rb") as f:
        params = pickle.load(f)

    model = NanoLM(
        input_vocab_size=config["input_vocab_size"],
        output_vocab_size=config["output_vocab_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        head_size=config["head_size"],
        dropout_rate=config["dropout_rate"],
        embed_size=config["embed_size"],
        block_size=config["block_size"],
    )
    data_filename = f"data/{data_name}/input.txt"
    model_path = f"model/{data_name}"

    os.makedirs(model_path, exist_ok=True)
    # platform check
    print("JAX running on", jax.devices()[0].platform.upper())

    with open(data_filename, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    input_tokenizer = TwoCharTokenizer()
    input_tokenized_text = jnp.array(input_tokenizer.encode(text))
    print("text:", text[:100])
    print("encoded:", input_tokenized_text[:10])
    print("decoded:", input_tokenizer.decode(input_tokenized_text[:10].tolist()))

    output_tokenizer = WordTokenizer()
    output_tokenized_text = jnp.array(output_tokenizer.encode(text))
    print("text:", text[:100])
    print("encoded:", output_tokenized_text[:10])
    print("decoded:", output_tokenizer.decode(output_tokenized_text[:10].tolist()))

    train_ratio = 0.90
    total_tokens = len(input_tokenized_text)
    split_idx = int(train_ratio * total_tokens)
    train_data = (input_tokenized_text[:split_idx], output_tokenized_text[:split_idx])
    eval_data = (input_tokenized_text[split_idx:], output_tokenized_text[split_idx:])

    dynamic_slice_vmap = jax.vmap(jax.lax.dynamic_slice, in_axes=(None, 0, None))

    @jax.jit
    def get_accuracy(params, x, y):
        logits = model.apply(params, x, training=False)
        return jnp.mean(jnp.argmax(logits, axis=-1) == y)

    @jax.jit
    def get_all(random_key, data):
        """Prepares a random batch of training data.

        Args:
            random_key: A random seed for sampling a batch.
            data: The complete training dataset.

        Returns:
            x: Input sequences.
            y: Target sequences (shifted inputs).
        """
        ix = jnp.arange(0, len(data) - config["block_size"] + 1).reshape(-1, 1)
        x = dynamic_slice_vmap(data, ix, (config["block_size"],))

        return x

    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    X = get_all(subkey, eval_data[0])
    Y = get_all(subkey, eval_data[1])
    batch_size = 256
    test_accuracy = []
    for i in range(0, X.shape[0] - batch_size, batch_size):
        x = X[i : i + batch_size].reshape(batch_size, -1)
        y = Y[i : i + batch_size].reshape(batch_size, -1)
        acc = get_accuracy(params, x, y)
        test_accuracy.append(acc)

    test_accuracy = jnp.mean(jnp.array(test_accuracy))
    print(f"Test accuracy: {test_accuracy}")

    X = get_all(subkey, train_data[0])
    Y = get_all(subkey, train_data[1])
    train_accuracy = []
    for i in range(0, X.shape[0] - batch_size, batch_size):
        x = X[i : i + batch_size].reshape(batch_size, -1)
        y = Y[i : i + batch_size].reshape(batch_size, -1)
        acc = get_accuracy(params, x, y)
        train_accuracy.append(acc)
    train_accuracy = jnp.mean(jnp.array(train_accuracy))
    print(f"Train accuracy: {train_accuracy}")


if __name__ == "__main__":
    main()
