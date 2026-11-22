import unittest
import torch
from attention import CausalSelfAttention, MultiHeadAttention


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.embed_size = 256
        self.heads = 8
        self.dropout = 0.2
        self.batch_size = 4
        self.seq_length = 10

        self.attention = CausalSelfAttention(self.embed_size, self.heads, self.dropout)
        self.multi_head_attention = MultiHeadAttention(self.embed_size, self.heads, self.dropout)

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def test_causal_self_attention_forward(self):
        self.attention.eval()
        x = torch.randn((self.batch_size, self.seq_length, self.embed_size))
        out = self.attention(x, x, x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_length, self.embed_size))

    def test_multi_head_attention_forward(self):
        self.multi_head_attention.eval()
        values = torch.randn((self.batch_size, self.seq_length, self.embed_size))
        keys = torch.randn((self.batch_size, self.seq_length, self.embed_size))
        query = torch.randn((self.batch_size, self.seq_length, self.embed_size))
        out = self.multi_head_attention(values, keys, query)
        self.assertEqual(out.shape, (self.batch_size, self.seq_length, self.embed_size))

    def test_device_handling(self):
        device = self._get_device()
        self.attention.to(device).eval()
        self.multi_head_attention.to(device).eval()

        x = torch.randn((self.batch_size, self.seq_length, self.embed_size), device=device)
        out_causal = self.attention(x, x, x)
        out_mha = self.multi_head_attention(x, x, x)

        self.assertEqual(out_causal.device.type, device.type)
        self.assertEqual(out_mha.device.type, device.type)

    def test_causal_masking(self):
        self.attention.eval()
        torch.manual_seed(42)
        x1 = torch.randn((self.batch_size, self.seq_length, self.embed_size))
        x2 = x1.clone()
        x2[:, 5:, :] = torch.randn((self.batch_size, self.seq_length - 5, self.embed_size))

        out1 = self.attention(x1, x1, x1)
        out2 = self.attention(x2, x2, x2)

        self.assertTrue(torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5))
        self.assertFalse(torch.allclose(out1[:, 5:, :], out2[:, 5:, :], atol=1e-5))

    def test_scaling(self):
        self.multi_head_attention.eval()
        torch.manual_seed(123)
        x = torch.randn((self.batch_size, self.seq_length, self.embed_size))

        head_dim = self.embed_size // self.heads
        expected_scale = head_dim ** 0.5
        q = self.multi_head_attention.queries(
            x.reshape(self.batch_size, self.seq_length, self.heads, head_dim)
        )
        k = self.multi_head_attention.keys(
            x.reshape(self.batch_size, self.seq_length, self.heads, head_dim)
        )
        raw_energy = torch.einsum("nqhd,nkhd->nhqk", [q, k])
        scaled_energy = raw_energy / expected_scale

        self.assertTrue(torch.allclose(scaled_energy * expected_scale, raw_energy, atol=1e-5))

    def test_dropout(self):
        torch.manual_seed(99)
        x = torch.randn((self.batch_size, self.seq_length, self.embed_size))

        self.attention.train()
        out_a = self.attention(x, x, x)
        out_b = self.attention(x, x, x)
        self.assertFalse(torch.equal(out_a, out_b))

        self.attention.eval()
        out_eval_a = self.attention(x, x, x)
        out_eval_b = self.attention(x, x, x)
        self.assertTrue(torch.equal(out_eval_a, out_eval_b))


if __name__ == "__main__":
    unittest.main()
