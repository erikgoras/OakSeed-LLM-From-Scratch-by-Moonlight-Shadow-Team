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
- **Agent Workflow:**
- **Challenges & Fixes:**
- **Verification:** `python3 -m unittest discover -s tests -v`
