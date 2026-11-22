
# Lab Notes: OakSeed LLM From Scratch (by Moonlight Shadow Team)

**Project:** `OakSeed-LLM-From-Scratch-by-Moonlight-Shadow-Team`  
**Philosophy:** First principles, one module at a time, minimal dependencies.

---

## Module 0 — Setup & Architecture Contract
- **Goal:** Establish repository structure (`data/`, `src/`, `tests/`) and `SPEC.md`.
- **Execution:** Created clean project workspace on Mac Magnum alongside the local n8n + Ollama Moonlight Shadow orchestration stack.
- **Takeaway:** Separating the agent team's control plane from the target repository keeps contracts clean and prevents context clutter.

## Module 1 — Data & Character Tokenizer (`src/tokenizer.py`)
- **Goal:** Build a deterministic character-level tokenizer and train/val splitter using only the Python standard library.
- **Status:** Completed and pushed to `main`.
- **Verification:** `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v`

## Module 2 — Causal Self-Attention & Multi-Head Attention (`src/attention.py`)
- **Goal:** Implement scaled dot-product causal self-attention and multi-head attention with `(B, T, C)` shape preservation, `mps`/`cpu` device handling, causal masking, scaling (`d_k ** -0.5`), and dropout.
- **Status:** Completed and pushed to `main` (`6d50cf8`), including `tests/test_attention.py` and `tests/__init__.py`.
- **Challenges & Fixes:** Added `tests/__init__.py` and ran discovery with `PYTHONPATH=src` and `-t .` for Python 3.14 test discovery, then aligned `src/attention.py` and `tests/test_attention.py` with `SPEC.md`.
- **Verification:** `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -t . -v` (10/10 tests passing).

---

## Remaining Modules & Next Steps (3 steps left)
1. **Module 3 — `src/model.py` (Next):** Transformer Block & GPT Language Model + `tests/test_model.py`.
2. **Module 4 — `src/train.py`:** Training loop, validation loss & checkpoints + tests.
3. **Module 5 — `src/generate.py`:** Autoregressive text generation (temperature / top-k) + tests.
