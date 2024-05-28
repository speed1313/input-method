import re

# load shakespere.txt
with open("shakespeare.txt") as f:
    text = f.read()
# text = text.lower()
text = re.sub(r"[^a-zA-Z]", " ", text)
text = re.sub(r"\s+", " ", text)
text = text[:10000]
print(text[:100])

# test (two characters, one word) pairs data

print("======Test data=======")
for word in text.split()[:10]:
    print(word[:2], word)
