import unittest
from pathlib import Path
from src.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.test_file = Path('tests/test_input.txt')
        with open(self.test_file, 'w', encoding='utf-8') as f:
            f.write("Hello, world!")
        self.tokenizer = Tokenizer(self.test_file)
        self.tokenizer.build_vocab()

    def test_build_vocab(self):
        expected = {
            ' ': 0, '!': 1, ',': 2, 'H': 3, 'd': 4, 'e': 5,
            'l': 6, 'o': 7, 'r': 8, 'w': 9
        }
        self.assertEqual(self.tokenizer.char_to_idx, expected)

    def test_encode(self):
        encoded = self.tokenizer.encode("Hello, world!")
        expected = [3, 5, 6, 6, 7, 2, 0, 9, 7, 8, 6, 4, 1]
        self.assertEqual(encoded, expected)

    def test_decode(self):
        decoded = self.tokenizer.decode([3, 5, 6, 6, 7, 2, 0, 9, 7, 8, 6, 4, 1])
        expected = "Hello, world!"
        self.assertEqual(decoded, expected)

    def test_split_dataset(self):
        train_data, val_data = self.tokenizer.split_dataset(0.8)
        # len("Hello, world!") = 13 chars (including spaces)
        # cutoff = int(13 * 0.8) = 10
        self.assertEqual(len(train_data), 10)
        self.assertEqual(len(val_data), 3)

    def tearDown(self):
        self.test_file.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main()
