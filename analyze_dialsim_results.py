import argparse
import json
import os
import statistics
from typing import Dict, List, Optional, Tuple

import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F


def simple_tokenize(text: str) -> List[str]:
    text = str(text or "")
    return text.lower().replace(".", " ").replace(",", " ").replace("!", " ").replace("?", " ").split()


def calc_token_set_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_BLEU_SMOOTH = SmoothingFunction().method1


def calc_rouge(prediction: str, reference: str) -> Dict[str, float]:
    scores = _ROUGE.score(reference, prediction)
    return {
        "rougeL_f": float(scores["rougeL"].fmeasure),
        "rouge2_f": float(scores["rouge2"].fmeasure),
    }


def calc_bleu1(prediction: str, reference: str) -> float:
    try:
        pred_tokens = nltk.word_tokenize(str(prediction or "").lower())
        ref_tokens = [nltk.word_tokenize(str(reference or "").lower())]
        return float(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=_BLEU_SMOOTH))
    except Exception:
        return 0.0


def calc_meteor(prediction: str, reference: str) -> float:
    try:
        return float(meteor_score([str(reference or "").split()], str(prediction or "").split()))
    except Exception:
        return 0.0


def batched_sbert_similarity(preds: List[str], refs: List[str], batch_size: int = 128) -> List[float]:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    sims: List[float] = []
    for i in range(0, len(preds), batch_size):
        p = preds[i : i + batch_size]
        r = refs[i : i + batch_size]
        with torch.no_grad():
            e1 = model.encode(p, convert_to_tensor=True, device=device)
            e2 = model.encode(r, convert_to_tensor=True, device=device)
            e1 = F.normalize(e1, p=2, dim=1)
            e2 = F.normalize(e2, p=2, dim=1)
            batch_sims = (e1 * e2).sum(dim=1).detach().cpu().numpy().tolist()
            sims.extend([float(x) for x in batch_sims])
    return sims


def _mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def load_records(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def extract_retrieval_proxy_stats(records: List[dict]) -> Dict[str, float]:
    """Compute retrieval proxy stats from records (best-effort; supports both HiGMem and A-Mem outputs)."""
    precisions = []
    recalls = []
    ks = []

    for r in records:
        trace = r.get("retrieval_trace") or {}
        # HiGMem: relevant_turn_ids (list[str]) + evidence_turn_ids_by_answer_string (list[str])
        if isinstance(trace, dict) and "relevant_turn_ids" in trace and "evidence_turn_ids_by_answer_string" in r:
            retrieved = trace.get("relevant_turn_ids") or []
            evidence = r.get("evidence_turn_ids_by_answer_string") or []
        # A-Mem: context_indices_with_duplicates (list[int]) + evidence_indices_by_answer_string (list[int])
        elif isinstance(trace, dict) and "context_indices_with_duplicates" in trace and "evidence_indices_by_answer_string" in r:
            retrieved = trace.get("context_indices_with_duplicates") or []
            evidence = r.get("evidence_indices_by_answer_string") or []
        else:
            continue

        if not isinstance(retrieved, list) or not isinstance(evidence, list):
            continue

        k = len(retrieved)
        ks.append(float(k))

        # skip empty evidence to match LoCoMo recall convention
        if len(evidence) == 0:
            continue

        retrieved_set = set(retrieved)
        evidence_set = set(evidence)
        hits = len(retrieved_set & evidence_set)
        precisions.append(hits / k if k > 0 else 0.0)
        recalls.append(hits / len(evidence_set) if len(evidence_set) > 0 else 0.0)

    return {
        "avg_k": _mean(ks),
        "precision_proxy_macro": _mean(precisions),
        "recall_proxy_macro": _mean(recalls),
        "num_questions_with_nonempty_evidence": float(len(precisions)),
    }


def analyze(path: str, compute_sbert: bool) -> Dict[str, float]:
    records = load_records(path)
    preds = [str(r.get("prediction", "") or "") for r in records]
    refs = [str(r.get("reference", "") or "") for r in records]

    f1s = [calc_token_set_f1(p, r) for p, r in zip(preds, refs)]
    bleu1s = [calc_bleu1(p, r) for p, r in zip(preds, refs)]
    rouges = [calc_rouge(p, r) for p, r in zip(preds, refs)]
    rougeL = [x["rougeL_f"] for x in rouges]
    rouge2 = [x["rouge2_f"] for x in rouges]
    meteors = [calc_meteor(p, r) for p, r in zip(preds, refs)]

    out: Dict[str, float] = {
        "num_questions": float(len(records)),
        "f1_mean": _mean(f1s),
        "bleu1_mean": _mean(bleu1s),
        "rougeL_mean": _mean(rougeL),
        "rouge2_mean": _mean(rouge2),
        "meteor_mean": _mean(meteors),
    }

    # latency
    durs = []
    for r in records:
        v = _safe_float(r.get("duration_seconds"))
        if v is not None:
            durs.append(v)
    out["avg_time_seconds"] = _mean(durs)

    # retrieval proxy stats (if present)
    out.update(extract_retrieval_proxy_stats(records))

    if compute_sbert:
        sims = batched_sbert_similarity(preds, refs)
        out["sbert_similarity_mean"] = _mean(sims)
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze DialSim predictions JSONL and print Table-2 style metrics.")
    parser.add_argument("--input", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--name", type=str, default=None, help="Optional run name label.")
    parser.add_argument("--compute_sbert", action="store_true", help="Compute SBERT similarity (can be slow).")
    parser.add_argument("--scale_100", action="store_true", help="Multiply metrics by 100 to match A-Mem paper table.")
    args = parser.parse_args()

    res = analyze(args.input, compute_sbert=args.compute_sbert)

    scale = 100.0 if args.scale_100 else 1.0
    table = {
        "Method": args.name or os.path.basename(args.input),
        "F1": res["f1_mean"] * scale,
        "BLEU-1": res["bleu1_mean"] * scale,
        "ROUGE-L": res["rougeL_mean"] * scale,
        "ROUGE-2": res["rouge2_mean"] * scale,
        "METEOR": res["meteor_mean"] * scale,
        "SBERT Similarity": (res.get("sbert_similarity_mean", 0.0) * scale) if args.compute_sbert else None,
        "N": int(res["num_questions"]),
        "Avg QA time (s)": res["avg_time_seconds"],
        "Avg K": res.get("avg_k", 0.0),
        "Precision@K (proxy)": res.get("precision_proxy_macro", 0.0),
        "Recall@K (proxy)": res.get("recall_proxy_macro", 0.0),
        "N (nonempty evidence)": int(res.get("num_questions_with_nonempty_evidence", 0.0)),
    }

    print(json.dumps({"raw": res, "table": table}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
