from pathlib import Path

class Tokenizer:
    def __init__(self, input_file: Path):
        self.input_file = input_file
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.data = []

    def build_vocab(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))  # Sort by ASCII/Unicode value
        for idx, char in enumerate(chars):
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
        self.data = [self.char_to_idx[char] for char in text]

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[char] for char in text]

    def decode(self, tokens: list[int]) -> str:
        return ''.join(self.idx_to_char[token] for token in tokens)

    def split_dataset(self, train_ratio: float = 0.8):
        cutoff = int(len(self.data) * train_ratio)
        return self.data[:cutoff], self.data[cutoff:]

#
