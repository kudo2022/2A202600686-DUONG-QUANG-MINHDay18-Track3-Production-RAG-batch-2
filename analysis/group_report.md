# Group Report - Lab 18: Production RAG

**Group:** Individual submission  
**Date:** 2026-06-22

## Member and Module Ownership

| Name | Module | Status | Verification |
|------|--------|--------|--------------|
| Duong Quang Minh | M1: Chunking | Done | `pytest tests/test_m1.py` |
| Duong Quang Minh | M2: Hybrid Search | Done | `pytest tests/test_m2.py` |
| Duong Quang Minh | M3: Reranking | Done | `pytest tests/test_m3.py` |
| Duong Quang Minh | M4: Evaluation | Done | `pytest tests/test_m4.py` |
| Duong Quang Minh | M5: Enrichment | Done | `pytest tests/test_m5.py` |

## Overall Verification

- Full test suite: `37/37` passed
- End-to-end script: `python main.py` completed successfully
- Reports generated: `reports/naive_baseline_report.json`, `reports/ragas_report.json`

## RAGAS Results

| Metric | Naive | Production | Delta |
|--------|-------|------------|-------|
| Faithfulness | 0.4519 | 0.5112 | +0.0593 |
| Answer Relevancy | 0.2407 | 0.2686 | +0.0279 |
| Context Precision | 0.1768 | 0.1900 | +0.0132 |
| Context Recall | 0.1417 | 0.1812 | +0.0395 |

## Key Findings

1. **Biggest improvement:** Faithfulness and context recall improved the most after moving from basic paragraph chunks to hierarchical chunks, hybrid retrieval, reranking, and chunk enrichment.
2. **Biggest challenge:** The environment had multiple real-world blockers, including a wrong OpenAI placeholder key, Windows console Unicode issues, and no running Docker daemon for Qdrant.
3. **Surprise finding:** Even with fallback evaluation and fallback enrichment, the production pipeline still outperformed the naive baseline across all four metrics.

## Presentation Notes

1. The production pipeline beat the naive baseline on every tracked metric, though all absolute scores are still below a strong production target.
2. The most valuable technical upgrade was retrieval quality, especially hierarchical chunking plus BM25+dense fusion and reranking.
3. The most useful failure pattern was low `context_recall` on numeric and approval-threshold questions, which points to retrieval coverage rather than generation quality.
4. Next optimization would focus on metadata-aware retrieval, overlap chunking for numeric facts, and running with a valid OpenAI key plus Docker-backed Qdrant.
