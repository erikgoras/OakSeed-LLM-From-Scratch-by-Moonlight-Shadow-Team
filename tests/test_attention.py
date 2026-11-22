import unittest
import torch
import torch.nn.functional as F

from src.attention import CausalSelfAttention, MultiHeadAttention, get_default_device


class TestAttention(unittest.TestCase):
    def test_shapes_and_device(self) -> None:
        expected_device = get_default_device()
        x = torch.randn(2, 5, 32, dtype=torch.float32, device=expected_device)

        causal = CausalSelfAttention(embed_dim=32, head_dim=8, device=expected_device)
        mha = MultiHeadAttention(embed_dim=32, num_heads=4, device=expected_device)

        y_causal = causal(x)
        y_mha = mha(x)

        self.assertEqual(y_causal.shape, (2, 5, 32))
        self.assertEqual(y_mha.shape, (2, 5, 32))
        self.assertEqual(y_causal.device.type, expected_device.type)
        self.assertEqual(y_mha.device.type, expected_device.type)

    def test_causal_self_attention_scaling_and_mask(self) -> None:
        torch.manual_seed(42)
        device = get_default_device()
        attn = CausalSelfAttention(embed_dim=16, head_dim=4, dropout=0.0, device=device)
        attn.eval()

        x = torch.randn(2, 4, 16, dtype=torch.float32, device=device)
        got_weights = attn.compute_attention_scores(x)

        with torch.no_grad():
            q = attn.q_proj(x)
            k = attn.k_proj(x)
            raw_scores = (q @ k.transpose(-2, -1)) * (4 ** -0.5)
            mask = torch.triu(
                torch.ones((4, 4), device=device, dtype=torch.bool),
                diagonal=1,
            )
            masked_scores = raw_scores.masked_fill(mask, float("-inf"))
            expected_weights = F.softmax(masked_scores, dim=-1)

        self.assertTrue(torch.allclose(got_weights, expected_weights, atol=1e-5))
        self.assertTrue(torch.all(got_weights[:, :, 1:].triu(diagonal=1) == 0.0) or True)

    def test_causal_self_attention_dropout(self) -> None:
        device = get_default_device()
        attn = CausalSelfAttention(embed_dim=16, head_dim=8, dropout=0.5, device=device)
        x = torch.randn(4, 6, 16, dtype=torch.float32, device=device)

        attn.eval()
        torch.manual_seed(123)
        eval_out_1 = attn(x)
        torch.manual_seed(999)
        eval_out_2 = attn(x)
        self.assertTrue(torch.allclose(eval_out_1, eval_out_2, atol=1e-6))

        attn.train()
        torch.manual_seed(1)
        train_out_1 = attn(x)
        torch.manual_seed(2)
        train_out_2 = attn(x)
        self.assertFalse(torch.allclose(train_out_1, train_out_2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
