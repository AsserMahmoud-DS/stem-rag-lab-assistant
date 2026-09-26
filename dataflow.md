# Dataflow — Iteration 1

End-to-end flow from ECE PDFs → judged answers across four RAG methods.
Mirrors AGENTS.md §14. Cross-component links live in `docs/references.md`,
not inline in code.

## 1. Overview

```
                            ┌───────────────────────┐
                            │   data/*.pdf  (×8)    │
                            └──────────┬────────────┘
                                       │  OpenDataLoader hybrid
                                       │  (local + Docling backend)
                                       ▼
                            ┌───────────────────────┐
                            │  loaded_data/*.json   │   element tree:
                            │  (per document)       │   {paragraph, heading,
                            └──────────┬────────────┘    caption, table, list,
                                       │                 image, …} + bbox
                                       │  + page
                                       ▼
                         ┌──────────────────────────┐
                         │  corpus.chunking         │  per
                         │  (_chunk_single_doc)     │  tests/test_chunking_contract.py
                         └─────┬──────────────┬─────┘  (chunk_id, doc_id,
                               │              │       page_number, order, …)
               caption text +   │              │
               ai_description   ▼              ▼
               embedded into  ┌─────┐      ┌──────────┐
               linked chunk   │chunks│      │ images   │  image_id ──┐
                              │.json │      │ .json    │  linked_     ├─→ caption_text +
                              └──┬───┘      └────┬─────┘  chunk_id     │   ai_description
                                 │               │                     │   embedded into
                                 │               │                     │   that chunk's text
                    bge-m3      │               │  (deterministic
                    dense       │               │   via OpenDataLoader
                    embed       │               │   "linked content id")
                                ▼               │
                     ┌────────────────────┐     │
                     │ embed cache        │     │
                     │ (persisted)        │     │
                     └────────────────────┘     │
                                                 │
   ╔═════════════════════════════════════════════╧═══════════════════╗
   ║  GRAPH CONSTRUCTION (ours — runs once per corpus version)       ║
   ║                                                                  ║
   ║  for each chunk in chunks.json:                                  ║
   ║      Groq LLM (gpt-oss-20b, ECE-tuned entity-type prompt)        ║
   ║        → (entities[], relations[]) in JSON                        ║
   ║  → entity normalization (case-insensitive + alias merge)         ║
   ║  → relation dedup                                                ║
   ║  → graph.json = {entities, relations, chunk↔entity links}         ║
   ║  (no communities, no gleaning, full rebuild on corpus change)    ║
   ╚════════════════════════════════════════════════════════════════╝
```

Storage at this point is all local JSON + LlamaIndex `SimpleVectorStore`
(in-memory, persisted to disk). No Postgres in iter-1.

**Hybrid backend:** `opendataloader-pdf-hybrid --port 5002
--enrich-picture-description --enrich-formula`. Routes complex pages to
Docling for table extraction, formula LaTeX, and picture description.

## 2. Query-time flow (per method)

All five methods run through the custom eval runner (`evaluation/run_eval.py`),
which logs retrieved context, answer, and latency per run and judges each cell
with the custom Groq LLM-judge (`evaluation/judge.py`). A thin sidecar
run-log additionally records `attached_image_ids` per (question, method)
because the judge is unaware of the `images.json` layer.

```
                    query (1 of 40 Qs)              golden answer ↘
                          │                            (for judge)
   ┌──────────────────────┼────────────────────────┐
   │                      ▼                          │
   │   method.choose:  ┌─────────┐  ┌─────────┐    │
   │   ┌──────────┐    │ vector  │  │ BM25    │    │
   │   │  naive   │    │ top-k   │  │ top-k   │    │
   │   └──────────┘    └────┬────┘  └────┬────┘    │
   │                         │            │         │
   │   ┌──────────┐          ▼            ▼         │
   │   │ hybrid   │    QueryFusionRetriever(RRF)     │
   │   └──────────┘    (no rerank in iter-1)          │
   │                         │ top-k fused chunks     │
   │                                                 │
   │   ┌──────────┐    seeds (BM25 ∪ vector top-k)   │
   │   │  ours:   │      → map seeds → entities      │
   │   │ h+graph  │      → one-hop over entity +     │
   │   └──────────┘        relation neighbours       │
   │                        (cap 2–4 per seed)       │
   │      → neighbour chunks (deduped)               │
   │                                                 │
   │   ┌──────────┐  lightrag-hku==<pinned>          │
   │   │  vanilla │  QueryParam(mode="mix",          │
   │   │ LightRAG │          only_need_context=True) │
   │   └──────────┘  → retrieved chunks (their ids)  │
   ╔═══════════════════╧══════════════════════════╗
   ║ image attach (IDENTICAL for every method):  ║
   ║   for chunk_id in retrieved_chunks:           ║
   ║       images where linked_chunk_id == chunk  ║
   ║   → caption_text + ai_description in context; src_path in refs ║
   ╚═══════════════════╤══════════════════════════╝
                       │
                       ▼
            ┌────────────────────────────┐
            │  shared Groq answer prompt │  openai/gpt-oss-20b
            │  (context + query)         │  (same for every method)
            └─────────────┬──────────────┘
                          │
                          ▼
                       answer
                       (+ image refs side payload for the UI)
                          │
                          ▼
            ┌────────────────────────────┐
            │  custom Groq LLM-judge     │  judged by
            │  Groundedness /           │  openai/gpt-oss-120b on Groq
            │  AnswerRelevance /        │  (via the shared Groq client —
            │  ContextRelevance /       │   no TruLens)
            │  GroundTruthAgreement      │
            └─────────────┬──────────────┘
                          ▼
                 per-question per-method scores
                          │
                          ▼
            ┌────────────────────────────┐
            │  aggregate: per-category   │  40/20/20/10/10
            │  mean + overall mean       │  cross-doc / multi-hop /
            │  + latency / cost table   │  same-doc / lookup / paraphrase
            └────────────────────────────┘
```

## 3. Design invariants (why the rows above look the way they do)

- **Same synthesis prompt across all five methods** — the *only* variable
  between methods is *which chunks reach the prompt*. This is the
  fairness guarantee (AGENTS.md §14.2). Vanilla-LightRAG uses
  `only_need_context=True` precisely so its chunk set can be routed
  through the shared prompt instead of being answered by LightRAG's own
  synthesizer.
- **No communities, no high/low keyword LLM at query time** in our
  hybrid+graph method (the contribution). Seeds come from BM25 ∪ vector
  chunk retrieval, then expand one hop over the graph. Vanilla LightRAG
  keeps its own dual-keyword retrieval so it behaves as its authors
  intended — that difference is part of what we compare.
- **System-vs-system, not policy-isolation.** Our graph and LightRAG's
  graph are different graphs (different prompts/merge), so the comparison
  is "our pipeline vs LightRAG's pipeline", not "same graph, different
  retrieval". Cleaner policy isolation is a possible iter-2 ablation.
- **Image linkage is deterministic**, not a similarity guess: OpenDataLoader
  links a `caption` element to its `image` via `linked content id`; we map
  that caption into the enclosing chunk. Images with an `ai_description` field
  (from Docling) are also linked. A figure is returned iff its caption-bearing
  or ai_description-bearing chunk is retrieved (stated limitation in §14.4).
- **Incremental at the chunk layer** (satisfies §13): unchanged `doc_id`s
  are not re-embedded or re-upserted; only new/changed chunks enter
  `chunks.json` + the embed cache. The **graph** is a full rebuild in
  iter-1 — incremental graph merge is iter-2.

## 4. Local artifacts once iter-1 is wired

```
stem_rag_lab_assistant/
├── data/                       ← source PDFs (8)
├── loaded_data/                ← OpenDataLoader JSON per PDF
├── chunks.json                 ← chunk_id, doc_id, text, embedding, metadata
├── images.json                 ← image_id, doc_id, page, bbox, src, caption, ai_description, linked_chunk_id
├── graph.json                  ← entities, relations, chunk↔entity links  (ours)
├── lightrag_data/              ← isolated working_dir for the throwaway vanilla baseline + lightrag_hybrid
├── data/questions_dataset/dataset.json ← 40 Qs + golden answers + category + source_chunk_ids
├── src/stem_rag_lab_assistant/evaluation/results_*.json  ← per-method judged results
├── src/stem_rag_lab_assistant/evaluation/comparison.json ← aggregated scores
└── runs/runs_sidecar.jsonl     ← (question, method) → retrieved_chunks + attached_image_ids
```

The `lightrag_data/` directory is written only by the pinned `lightrag-hku`
baseline; our own pipeline does not read from it (we read our own `graph.json`).
`lightrag_hybrid` reads its persisted graph artifacts read-only. It exists for
reproducibility of the baseline numbers only.