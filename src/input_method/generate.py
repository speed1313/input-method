import jax
from model import NanoLM
import click
from tokenizer import TwoCharTokenizer, WordTokenizer
import pickle
import json
import jax.numpy as jnp


@click.command()
@click.option("--data_name", type=str, default="shakespeare")
@click.option("--prompt", type=str, default="私は")
@click.option("--max_new_tokens", type=int, default=60)
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_k", type=int, default=25)
def main(
    data_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    tokenizer_path = f"data/{data_name}/tokenizer.json"
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
    key = jax.random.PRNGKey(0)

    input_tokenizer = TwoCharTokenizer()
    output_tokenizer = WordTokenizer()

    text = "My name is Taro. I am a student."
    encoded_text = jnp.array(input_tokenizer.encode(text)).reshape(1, -1)
    output = []
    print(encoded_text)
    for i in range(encoded_text.shape[1]):
        print(encoded_text[:, : i + 1])
        output_token = model.apply(params, encoded_text[:, : i + 1], training=False)
        top_logits = output_token[:, -1, :]
        decoded = output_tokenizer.decode(top_logits.argmax(axis=-1).tolist())
        print(decoded)
        output.append(decoded)

    print(output)


if __name__ == "__main__":
    main()
