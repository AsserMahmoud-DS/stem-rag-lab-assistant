# Iter-1 Reproducible Snapshot (`iter1-ship`)

This directory snapshot freezes the **iteration-1** state of the STEM RAG Lab
Assistant: code (`main`), corpus artifacts, indices, and evaluation results.
It exists so iter-2 can regenerate working files without losing iter-1.

**Frozen branch:** `iter1` (tag `iter1-ship`). **Trunk:** `main` = iter-1 code
only (no heavy artifacts). **Active:** `iter2`.

---

## Evaluation summary (TruLens 0--1)

4 methods × 40 questions. Same answer model (`openai/gpt-oss-20b`), same judge
(`openai/gpt-oss-120b` via TruLens), same embedding (`BAAI/bge-m3`).

| Method | Groundedness | Answer Rel. | Context Rel. | GT Agreement | Mean | Avg. Chunks | Fallback |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| naive | 0.56 | 0.91 | 0.71 | 0.69 | **0.72** | 6.0 | 2 |
| hybrid | 0.53 | 0.93 | 0.72 | 0.70 | **0.72** | 4.6 | 3 |
| lightrag | 0.56 | 0.93 | 0.78 | 0.71 | **0.74** | 10.0 | 3 |
| lightrag_hybrid | 0.54 | 0.89 | 0.74 | 0.71 | **0.72** | 8.6 | 3 |

Capacity fallback: 11 empty answers rescued (naive 2, hybrid 3, lightrag 3,
lightrag_hybrid 3), **0 unresolved**. Same model, higher completion budget
(`max_completion_tokens=8192`, ≤4 attempts), reasoning unchanged.

Category note: LightRAG wins multi-hop + same-document; LightRAG-Hybrid wins
cross-document + lookup + paraphrase.

---

## Frozen configuration (from `comparison.json`)

```json
{
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "answer_model": "openai/gpt-oss-20b",
  "judge_model": "openai/gpt-oss-120b",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "vector_top_k": 6,
  "bm25_top_k": 6,
  "rerank_top_n": 4,
  "lightrag_chunk_top_k": 10,
  "answer_fallback_max_tokens": 8192,
  "answer_fallback_max_attempts": 4,
  "chunk_size": 512,
  "chunk_overlap": 80,
  "graph_neighbour_cap": 2,
  "graph_expansion_depth": 1,
  "graph_seed_entities_cap": 5,
  "graph_max_expanded_chunks": 10,
  "extraction_llm_model": "qwen/qwen3.8-27b",
  "extraction_max_concurrent": 2,
  "extraction_max_entities_per_chunk": 30,
  "extraction_max_total_per_chunk": 50,
  "extraction_max_tokens": 4096,
  "extraction_retry_max": 2
}
```

Schema version: `iter1`.

---

## Corpus

- 8 PDFs (4 lecture chapters + 4 labs), 150 chunks total, 62 figures (all linked).
- AI figure descriptions are **not** merged into chunk text (removed this iter).
- LightRAG graph: **754 entities, 976 relations**, built over the 150 chunks.

Per-document chunks: `chapter-1 Error in measurement` 25, `Lab1` 12,
`chap 3 DC&AC BRIDGES` 37, `Lab2` 2, `chapter(4) oscilloscope` 11, `Lab3` 4,
`chapter 5  ADC and DAC` 54, `Lab4` 5.

---

## Manifest (sha256)

```
c1a79361c25642cc86c7f5d11b3610e1a0e8833a2e40214731c89f46d1beed40  chunks.json
d30fb6734aa15e684c32ced6e272a9c89c8fb3583dcd8f1e00779adc01753da9  images.json
5f930c56a7964ed1ef52cda797a523517c69979f64229f969701b2b3423d90f4  lightrag_data/graph_chunk_entity_relation.graphml
76c7d5dd7fdca4113e6adc239df18bbfe42bb704d10f67d03241e6904dcacc2d  dataset/questions_dataset/dataset.json
1327c0266d242f4452cd40d4f39da506520d84cbfc8dbcbe8691e7a50bab6336  runs/runs_sidecar.jsonl
657af137bb5e327c669ebd514f4435c09c30d27a0b37fcc79fcb98b7f5b36b3a  reports/system_report.md
facbb43eb3e4620936f433565248131b5292e6893f97b43f684e73dbc8fd2b53  src/stem_rag_lab_assistant/evaluation/comparison.json
61bf16294eebb02db48466fe8f539e703d2789ef6eda963f5eb3a54b346d575c  src/stem_rag_lab_assistant/evaluation/results_naive.json
fdbb72aa22ef7765e90c092bc16f712f21cd260015ade4038c3a7ed791c29a02  src/stem_rag_lab_assistant/evaluation/results_hybrid.json
9a6dda68e5627f6385dde45e9430a5f8db9321bbc2f4766904f7f329bd04bbdd  src/stem_rag_lab_assistant/evaluation/results_lightrag.json
712dcad3e326cdb92a3126baa8f9539cea3384162455d69349744bb61f826ea4  src/stem_rag_lab_assistant/evaluation/results_lightrag_hybrid.json
```

Also included (not hashed above): `lightrag_data/` (full working dir) and
`dataset/` (raw PDFs + `questions_dataset/`).

**Excluded:** `loaded_data/` (OpenDataLoader output — regenerable from the
`dataset/` PDFs), `storage/` (rebuildable from `chunks.json`), `docs/`,
`plans/`, secrets (`tokens.txt`, `.env`), `.venv`, HuggingFace cache,
`__pycache__`.

---

## Reproduction

Frozen (no recompute needed): `chunks.json`, `images.json`, `lightrag_data/`,
`dataset/`, `runs/runs_sidecar.jsonl`, and the `results_*` / `comparison.json`
outputs. `loaded_data/` is intentionally excluded; regenerate it from the PDFs
with the OpenDataLoader hybrid backend (see `corpus/load.py`) only if you need
to re-run ingest.

To rebuild local indices from the frozen chunks (no GPU) and re-aggregate:

```bash
# vector + BM25 indices from chunks.json (embeddings already present)
uv run python -m stem_rag_lab_assistant.index.vector_store
uv run python -m stem_rag_lab_assistant.index.bm25_store

# recompute comparison.json from frozen results (no LLM calls)
uv run python -m stem_rag_lab_assistant.evaluation.aggregate
```

To re-run the evaluation from scratch (LLM calls; cost), see `README.md`:
`run_eval` → `fallback` → `aggregate`.
