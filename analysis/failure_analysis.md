# Failure Analysis - Lab 18: Production RAG

**Group:** Individual submission  
**Author:** Duong Quang Minh

## RAGAS Scores

| Metric | Naive Baseline | Production | Delta |
|--------|----------------|------------|-------|
| Faithfulness | 0.4519 | 0.5112 | +0.0593 |
| Answer Relevancy | 0.2407 | 0.2686 | +0.0279 |
| Context Precision | 0.1768 | 0.1900 | +0.0132 |
| Context Recall | 0.1417 | 0.1812 | +0.0395 |

## Bottom-5 Failures

### #1
- **Question:** Muon mua thiet bi tri gia 55 trieu can ai phe duyet?
- **Expected:** Don hang tren 50.000.000 VND can Tong Giam doc (CEO) phe duyet.
- **Got:** Retrieved context likely missed the chunk that states the final approval threshold above 50 million, so the answer did not fully mention the CEO approval rule.
- **Worst metric:** `context_recall`
- **Error Tree:** Output not fully correct -> context not complete -> procurement threshold chunk missing -> answer lost the final approval detail.
- **Root cause:** Retrieval recall for procurement policy is still weak when the question mixes item price and approver role.
- **Suggested fix:** Add metadata filtering for procurement/admin documents, strengthen BM25 matching on `phe duyet`, `CEO`, `tren 50 trieu`, and consider overlap between child chunks.

### #2
- **Question:** Luong thu viec cua nhan vien Junior muc cao nhat la bao nhieu?
- **Expected:** Junior cao nhat la 20.000.000 VND/thang. Luong thu viec = 85% x 20.000.000 = 17.000.000 VND/thang.
- **Got:** The answer likely retrieved only one part of the rule, such as the salary band or the trial percentage, but not both pieces together to compute the final number.
- **Worst metric:** `context_recall`
- **Error Tree:** Output incomplete -> context missing a required numeric rule -> answer cannot finish the 85 percent calculation correctly.
- **Root cause:** Numeric multi-step questions require more than one supporting fact in the same context window.
- **Suggested fix:** Preserve salary tables and percentage rules in overlapping child chunks, and boost retrieval for chunks containing `Junior`, `thu viec`, and `%`.

### #3
- **Question:** Khi phat hien malware tren may, nhan vien co nen tu xu ly khong?
- **Expected:** Khong. Nhan vien tuyet doi khong duoc tu y xu ly malware; phai bao cao trong vong 1 gio qua helpdesk hoac hotline CNTT.
- **Got:** The pipeline likely retrieved security-related context, but not the most specific incident-response chunk, so the answer was less explicit about the prohibition and reporting path.
- **Worst metric:** `context_precision`
- **Error Tree:** Output only partly sharp -> relevant context mixed with extra IT chunks -> rerank did not isolate the exact incident rule.
- **Root cause:** The query vocabulary overlaps with many IT/security chunks, which introduces noise before answer generation.
- **Suggested fix:** Add metadata signals for incident response, reduce noisy top-k before generation, and prioritize chunks containing strong negation like `khong duoc`.

### #4
- **Question:** Co can kich hoat xac thuc da yeu to (MFA) khong?
- **Expected:** Co. Tat ca nhan vien bat buoc kich hoat MFA cho email, VPN va he thong noi bo theo password policy v2.0.
- **Got:** The answer likely captured the MFA topic but lost some precision around `bat buoc`, the scope of systems, or the versioned policy source.
- **Worst metric:** `context_precision`
- **Error Tree:** Output somewhat correct -> context relevant but noisy -> answer did not focus tightly on the current MFA requirement.
- **Root cause:** Multiple IT/password chunks share similar vocabulary, so hybrid retrieval still returns redundant context.
- **Suggested fix:** Add version-aware metadata, boost chunks containing `MFA`, `bat buoc`, `v2.0`, and deduplicate similar IT chunks before answer generation.

### #5
- **Question:** Phu cap an trua hang thang la bao nhieu?
- **Expected:** Phu cap an trua la 1.000.000 VND/thang, chi tra cung ky luong.
- **Got:** The answer likely found the allowance topic but missed the exact amount or omitted the payroll timing detail.
- **Worst metric:** `context_recall`
- **Error Tree:** Output lacks key number -> context not complete -> numeric allowance chunk not ranked high enough.
- **Root cause:** Short numeric queries are sensitive to chunk boundaries and exact lexical matching.
- **Suggested fix:** Keep monetary values intact inside child chunks, add overlap around allowance policies, and favor BM25 for exact token matches like `an trua`.

## Case Study

**Selected question:** Muon mua thiet bi tri gia 55 trieu can ai phe duyet?

**Error Tree walkthrough:**
1. Output correct? -> Not fully.
2. Correct context retrieved? -> Not enough, because the key approval threshold chunk was missing.
3. Query rewrite good enough? -> Reasonable, but still too weak on role title and threshold language.
4. Best intervention point -> Retrieval recall, not generation.

**If I had one more hour, I would optimize:**
- Add metadata fields such as `category`, `version`, and `source`.
- Add overlap between hierarchical child chunks for numeric and policy-threshold facts.
- Save per-question answers and retrieved contexts into the report for deeper failure review.
