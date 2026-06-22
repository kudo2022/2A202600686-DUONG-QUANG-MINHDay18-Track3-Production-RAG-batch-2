# Reflection - Duong Quang Minh

## Part 1: Lecture Concept to Code Mapping

| Lecture Concept | Module | Function | Observation |
|-----------------|--------|----------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | Implemented sentence grouping with embedding-based similarity when available and lexical fallback otherwise. |
| Parent-child chunking | M1 | `chunk_hierarchical()` | Parent chunks preserve broader policy context while child chunks keep retrieval more precise. |
| BM25 + dense fusion | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Hybrid search improved all metrics over dense-only baseline, especially context recall. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Reranking helps push obviously relevant policy chunks above unrelated IT/HR chunks. |
| RAGAS evaluation | M4 | `evaluate_ragas()`, `failure_analysis()` | Added a heuristic fallback so evaluation still runs when no valid OpenAI key is available. |
| Contextual enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | Even fallback enrichment adds useful document framing before embedding. |

## Part 2: Difficulties and How I Solved Them

- **Dependency installation issue:** `ResolutionImpossible` and `regex` looked like a package conflict at first, but the real issue was that the terminal was not consistently using the repo `.venv`. I fixed this by installing and running everything through `.venv\\Scripts\\python.exe`.
- **Windows console issue:** `main.py` initially failed with `UnicodeEncodeError` while printing Unicode symbols. I fixed this by reconfiguring `stdout` and `stderr` to UTF-8 in the CLI entry points.
- **Invalid OpenAI key issue:** `.env` contained the placeholder `sk-...`, which triggered repeated `401` errors in enrichment and evaluation. I sanitized placeholder keys in `config.py` and skipped OpenAI-dependent paths when no valid key is present.
- **Torch compatibility warning:** installing dependencies upgraded `setuptools` too far for `torch 2.12.1`. I downgraded it to `<82`.
- **No Docker daemon:** `docker compose up -d` failed because Docker Desktop was not running. I kept a functional in-memory fallback path in dense search so the pipeline could still run end-to-end.

## Part 3: Action Plan for My Project

### Project: Vietnamese Internal Policy QA Assistant

### Current State

- Current RAG pipeline: basic retrieval and answer generation
- Known issues: weak retrieval recall on numeric facts, versioned policies, and approval-threshold questions

### Plan to Apply

1. [ ] **Chunking strategy:** use hierarchical chunking with slight overlap for tables, thresholds, and numeric rules.
2. [ ] **Search:** use hybrid BM25 + dense retrieval with Vietnamese segmentation to reduce vocabulary mismatch.
3. [ ] **Reranking:** keep a cross-encoder reranker for the top candidate set to improve precision before generation.
4. [ ] **Evaluation:** run RAGAS when a valid OpenAI key is available, and keep lightweight heuristic metrics for local testing.
5. [ ] **Enrichment:** add contextual prepend and metadata extraction first, then expand to summary/HyQA if latency budget allows.

### Timeline

- Week 1: stabilize ingestion, chunking, and hybrid retrieval
- Week 2: add reranking and metadata-aware retrieval
- Week 3: build evaluation set and compare naive vs production metrics
- Week 4: optimize failure cases involving versioning, numeric reasoning, and approval workflows
