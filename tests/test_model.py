import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from model import GPTLanguageModel, resolve_device


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.vocab_size = 64
        self.embed_dim = 32
        self.num_heads = 4
        self.num_layers = 2
        self.block_size = 16
        self.device = resolve_device()
        self.model = GPTLanguageModel(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            block_size=self.block_size,
            dropout=0.0,
            device=self.device,
        )
        self.model.eval()

    def test_output_shape_and_loss(self) -> None:
        batch_size, seq_len = 2, 8
        idx = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)
        targets = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=self.device)

        logits_only, loss_none = self.model(idx)
        self.assertEqual(logits_only.shape, (batch_size, seq_len, self.vocab_size))
        self.assertIsNone(loss_none)

        logits, loss = self.model(idx, targets=targets)
        self.assertEqual(logits.shape, (batch_size, seq_len, self.vocab_size))
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())

    def test_causality(self) -> None:
        seq_len = 8
        idx = torch.randint(0, self.vocab_size, (1, seq_len), device=self.device)
        idx_modified = idx.clone()
        last_val = int(idx_modified[0, -1].item())
        idx_modified[0, -1] = (last_val + 1) % self.vocab_size

        with torch.no_grad():
            logits, _ = self.model(idx)
            logits_modified, _ = self.model(idx_modified)

        self.assertTrue(
            torch.allclose(logits[:, :-1, :], logits_modified[:, :-1, :], atol=1e-4),
            "Earlier positions' logits changed when only the last token was modified.",
        )

    def test_sequence_length_validation(self) -> None:
        idx_too_long = torch.randint(
            0, self.vocab_size, (1, self.block_size + 1), device=self.device
        )
        with self.assertRaises(ValueError):
            self.model(idx_too_long)

    def test_embed_dim_validation(self) -> None:
        with self.assertRaises(ValueError):
            GPTLanguageModel(
                vocab_size=self.vocab_size,
                embed_dim=30,
                num_heads=4,
                num_layers=1,
                block_size=self.block_size,
                device="cpu",
            )

    def test_device_handling(self) -> None:
        self.model.to("cpu")
        self.assertEqual(next(self.model.parameters()).device.type, "cpu")

        if torch.backends.mps.is_available():
            self.model.to("mps")
            self.assertEqual(next(self.model.parameters()).device.type, "mps")


if __name__ == "__main__":
    unittest.main()
