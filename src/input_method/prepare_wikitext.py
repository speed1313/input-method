from datasets import load_dataset
import os

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

dir = "data/wikitext"
os.makedirs(dir, exist_ok=True)

with open(f"{dir}/input.txt", "w") as f:
    for example in ds["train"]:
        f.write(example["text"] + "\n")
