# Antigravity-Style Agent Harness

Modular, lightweight Python agent harness inspired by Google Antigravity primitives. Designed to run against OpenAI-compatible API endpoints (e.g., local `llama.cpp` or remote endpoints) with embedded SQLite vector memory and overwrite state planning.

## Features

1. **State-Based Planner & Context Guard (`harness/planner.py`)**:
   - Atomic `update_plan` overwrites `PLAN.md`.
   - Anti-repetition state guard and history context truncation.

2. **Embedded RAG Memory (`harness/memory.py`)**:
   - SQLite + `sqlite-vec` extension support (`vec_knowledge`, `knowledge_meta`).
   - Keyword and text search fallback.

3. **Workspace Security Guard (`harness/guard.py`)**:
   - Path containment check rejecting path traversal (`../`).

4. **Agent Core & Tool Registry (`harness/agent.py`, `harness/tools.py`)**:
   - OpenAI SDK integration with multi-turn `Plan -> Act -> Observe -> Reflect` loop.
   - Built-in primitives: `read_file`, `write_file`, `apply_patch`, `execute_command`, `update_plan`, `search_memory`.

## Testing

Run tests with `pytest`:

```bash
pytest -v
```
