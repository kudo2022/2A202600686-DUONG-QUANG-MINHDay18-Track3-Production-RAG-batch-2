from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, re
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, OPENAI_API_KEY


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _overlap_score(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if OPENAI_API_KEY:
        try:
            from ragas import evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
            from datasets import Dataset

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            df = result.to_pandas()
            per_question = [
                EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=list(row["contexts"]),
                    ground_truth=row["ground_truth"],
                    faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
                    answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
                    context_precision=float(row.get("context_precision", 0.0) or 0.0),
                    context_recall=float(row.get("context_recall", 0.0) or 0.0),
                )
                for _, row in df.iterrows()
            ]
            return {
                "faithfulness": _safe_mean([item.faithfulness for item in per_question]),
                "answer_relevancy": _safe_mean([item.answer_relevancy for item in per_question]),
                "context_precision": _safe_mean([item.context_precision for item in per_question]),
                "context_recall": _safe_mean([item.context_recall for item in per_question]),
                "per_question": per_question,
            }
        except Exception as e:
            print(f"  ⚠️  RAGAS evaluation failed: {e}")

    per_question: list[EvalResult] = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        combined_context = "\n".join(ctxs)
        faith = _overlap_score(answer, combined_context)
        answer_rel = max(_overlap_score(answer, question), _overlap_score(answer, ground_truth))

        if ctxs:
            ctx_prec = _safe_mean([max(_overlap_score(ctx, ground_truth), _overlap_score(ctx, question)) for ctx in ctxs])
        else:
            ctx_prec = 0.0
        ctx_recall = _overlap_score(combined_context, ground_truth)

        per_question.append(EvalResult(
            question=question,
            answer=answer,
            contexts=ctxs,
            ground_truth=ground_truth,
            faithfulness=faith,
            answer_relevancy=answer_rel,
            context_precision=ctx_prec,
            context_recall=ctx_recall,
        ))

    return {
        "faithfulness": _safe_mean([item.faithfulness for item in per_question]),
        "answer_relevancy": _safe_mean([item.answer_relevancy for item in per_question]),
        "context_precision": _safe_mean([item.context_precision for item in per_question]),
        "context_recall": _safe_mean([item.context_recall for item in per_question]),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or overstating beyond evidence", "Tighten the answer prompt and restrict output to retrieved context."),
        "context_recall": ("Missing relevant chunks during retrieval", "Improve chunking, enrichment, or hybrid retrieval coverage."),
        "context_precision": ("Too many irrelevant chunks retrieved", "Add reranking, metadata filters, or reduce retrieval breadth."),
        "answer_relevancy": ("Answer does not directly address the user question", "Refine the generation prompt and question-to-context alignment."),
    }

    ranked = []
    for item in eval_results:
        metrics = {
            "faithfulness": float(item.faithfulness),
            "answer_relevancy": float(item.answer_relevancy),
            "context_precision": float(item.context_precision),
            "context_recall": float(item.context_recall),
        }
        avg_score = _safe_mean(list(metrics.values()))
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    ranked.sort(key=lambda item: item["score"])
    return ranked[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
        "per_question": [
            asdict(item) if isinstance(item, EvalResult) else item
            for item in results.get("per_question", [])
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
