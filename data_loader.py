import json
import random
from typing import List, Tuple, Optional, Dict

class IntentDataset:
    def __init__(self, pairs: List[Tuple[str, str]], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x, y = self.pairs[idx]
        x_enc = self.tokenizer(x, truncation=True, max_length=self.max_length, return_tensors="pt")
        y_enc = self.tokenizer(y, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {"x": x, "y": y, "x_enc": x_enc, "y_enc": y_enc}

def load_jsonl(path: str, text_key: str) -> List[str]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append(obj[text_key])
    return data

def adversarial_transform(text: str, seed: Optional[int] = None) -> str:
    if seed is not None:
        random.seed(seed)
    tokens = text.split()
    if not tokens:
        return text
    t = random.choice(["shuffle", "drop", "repeat", "noise", "swap", "case"]) 
    if t == "shuffle":
        random.shuffle(tokens)
        return " ".join(tokens)
    if t == "drop":
        k = max(1, int(0.1 * len(tokens)))
        idxs = set(random.sample(range(len(tokens)), k))
        return " ".join([tok for i, tok in enumerate(tokens) if i not in idxs])
    if t == "repeat":
        i = random.randrange(len(tokens))
        tokens.insert(i, tokens[i])
        return " ".join(tokens)
    if t == "swap":
        i = random.randrange(len(tokens))
        j = random.randrange(len(tokens))
        tokens[i], tokens[j] = tokens[j], tokens[i]
        return " ".join(tokens)
    if t == "case":
        return " ".join([tok.upper() if random.random() < 0.2 else tok.lower() if random.random() < 0.2 else tok for tok in tokens])
    noise_chars = ["!", "?", ",", ".", "#", "$"]
    return "".join([c + (random.choice(noise_chars) if random.random() < 0.05 else "") for c in text])

def build_pairs(canonical_intents: List[str], seed: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs = []
    for y in canonical_intents:
        x = adversarial_transform(y, seed=seed)
        pairs.append((x, y))
    return pairs

def build_pairs_multi(canonical_intents: List[str], repeats: int = 1, seed: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs = []
    for y in canonical_intents:
        for r in range(repeats):
            s = None if seed is None else seed + r
            x = adversarial_transform(y, seed=s)
            pairs.append((x, y))
    return pairs