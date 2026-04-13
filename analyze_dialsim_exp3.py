import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def simple_tokenize(text: str) -> List[str]:
    text = str(text or "")
    return (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace("!", " ")
        .replace("?", " ")
        .split()
    )


def token_set_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def load_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_retrieved_and_evidence(record: dict) -> Optional[Tuple[List[Any], List[Any]]]:
    trace = record.get("retrieval_trace") or {}
    if not isinstance(trace, dict):
        return None

    # HiGMem: relevant_turn_ids + evidence_turn_ids_by_answer_string
    if "relevant_turn_ids" in trace and "evidence_turn_ids_by_answer_string" in record:
        retrieved = trace.get("relevant_turn_ids") or []
        evidence = record.get("evidence_turn_ids_by_answer_string") or []
        if isinstance(retrieved, list) and isinstance(evidence, list):
            return retrieved, evidence
        return None

    # A-Mem: context_indices_with_duplicates + evidence_indices_by_answer_string
    if "context_indices_with_duplicates" in trace and "evidence_indices_by_answer_string" in record:
        retrieved = trace.get("context_indices_with_duplicates") or []
        evidence = record.get("evidence_indices_by_answer_string") or []
        if isinstance(retrieved, list) and isinstance(evidence, list):
            return retrieved, evidence
        return None

    return None


@dataclass
class TokenAgg:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, usage: Dict[str, Any]) -> None:
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.total_tokens += int(usage.get("total_tokens") or 0)


def _get_scene_key(r: dict) -> str:
    # Prefer stable uid if present.
    uid = r.get("scene_uid")
    if uid:
        return str(uid)
    show = str(r.get("show", ""))
    ep = str(r.get("episode", ""))
    scene_id = str(r.get("scene_id", ""))
    return f"{show}|{ep}|{scene_id}"


def analyze(path: str) -> Dict[str, Any]:
    records = load_jsonl(path)

    # Distributions
    by_show = Counter()
    by_split = Counter()
    by_qtype = Counter()
    for r in records:
        by_show[str(r.get("show", ""))] += 1
        by_split[str(r.get("split", ""))] += 1
        by_qtype[str(r.get("q_type", ""))] += 1

    # F1
    f1s: List[float] = []
    f1_by_show: Dict[str, List[float]] = defaultdict(list)
    f1_by_split: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        p = str(r.get("prediction", "") or "")
        ref = str(r.get("reference", "") or "")
        f1 = token_set_f1(p, ref)
        f1s.append(f1)
        f1_by_show[str(r.get("show", ""))].append(f1)
        f1_by_split[str(r.get("split", ""))].append(f1)

    # QA token/time
    qa_tokens = TokenAgg()
    qa_durations: List[float] = []
    for r in records:
        usage = r.get("token_usage") or {}
        if isinstance(usage, dict):
            qa_tokens.add(usage)
        dur = _safe_float(r.get("duration_seconds"))
        if dur is not None:
            qa_durations.append(dur)

    # Memory construction (dedup by scene)
    seen_scenes: set = set()
    scenes_by_show = Counter()
    turns_total = 0
    construction_tokens = TokenAgg()
    construction_durations: List[float] = []
    for r in records:
        scene_key = _get_scene_key(r)
        if scene_key in seen_scenes:
            continue
        seen_scenes.add(scene_key)
        scenes_by_show[str(r.get("show", ""))] += 1
        scm = r.get("scene_memory_construction") or {}
        if isinstance(scm, dict):
            usage = scm.get("token_usage") or {}
            if isinstance(usage, dict):
                construction_tokens.add(usage)
            dur = _safe_float(scm.get("duration_seconds"))
            if dur is not None:
                construction_durations.append(dur)
            turns_total += int(scm.get("num_turns_added") or 0)

    # Effective turns_seen (includes scenes with 0 questions; derived from memory_state snapshots)
    turns_seen_max_by_show: Dict[str, int] = {}
    for r in records:
        show = str(r.get("show", ""))
        ms = r.get("memory_state") or {}
        if isinstance(ms, dict):
            ts = int(ms.get("turns_seen") or 0)
            turns_seen_max_by_show[show] = max(turns_seen_max_by_show.get(show, 0), ts)
    turns_seen_total = int(sum(turns_seen_max_by_show.values()))

    # Retrieval proxy stats
    ks: List[int] = []
    precisions: List[float] = []
    recalls: List[float] = []
    n_nonempty_evidence = 0
    for r in records:
        pair = _extract_retrieved_and_evidence(r)
        if not pair:
            continue
        retrieved, evidence = pair
        k = len(retrieved)
        ks.append(k)
        if len(evidence) == 0:
            continue
        n_nonempty_evidence += 1
        retrieved_set = set(retrieved)
        evidence_set = set(evidence)
        hits = len(retrieved_set & evidence_set)
        precisions.append(hits / k if k > 0 else 0.0)
        recalls.append(hits / len(evidence_set) if evidence_set else 0.0)

    out: Dict[str, Any] = {
        "path": path,
        "num_questions": len(records),
        "shows": dict(by_show),
        "splits": dict(by_split),
        "q_types": dict(by_qtype),
        "num_scenes": len(seen_scenes),
        "scenes_by_show": dict(scenes_by_show),
        # turns_total_observed only includes scenes that had >=1 question record in this file.
        # turns_total_effective comes from max(memory_state.turns_seen) per show and better matches
        # the actual amount of streaming construction performed (including scenes with 0 questions).
        "turns_total_observed": turns_total,
        "turns_total_effective": turns_seen_total,
        "turns_seen_max_by_show": dict(turns_seen_max_by_show),
        "metrics": {
            "f1_mean": _mean(f1s),
            "f1_by_show_mean": {k: _mean(v) for k, v in f1_by_show.items()},
            "f1_by_split_mean": {k: _mean(v) for k, v in f1_by_split.items()},
        },
        "retrieval": {
            "avg_k": _mean([float(x) for x in ks]),
            "precision_macro": _mean(precisions),
            "recall_macro": _mean(recalls),
            "num_questions_with_nonempty_evidence": n_nonempty_evidence,
        },
        "tokens": {
            "qa": {
                "prompt_tokens_total": qa_tokens.prompt_tokens,
                "completion_tokens_total": qa_tokens.completion_tokens,
                "total_tokens_total": qa_tokens.total_tokens,
                "avg_prompt_tokens_per_q": qa_tokens.prompt_tokens / len(records) if records else 0.0,
                "avg_completion_tokens_per_q": qa_tokens.completion_tokens / len(records) if records else 0.0,
                "avg_total_tokens_per_q": qa_tokens.total_tokens / len(records) if records else 0.0,
            },
            "construction": {
                "prompt_tokens_total": construction_tokens.prompt_tokens,
                "completion_tokens_total": construction_tokens.completion_tokens,
                "total_tokens_total": construction_tokens.total_tokens,
                "avg_total_tokens_per_turn_observed": (construction_tokens.total_tokens / turns_total) if turns_total > 0 else 0.0,
                # Best-effort: scale observed construction tokens/seconds to effective turns_total.
                # This assumes avg per-turn construction cost is roughly stable across scenes.
                "total_tokens_total_est_full": (
                    int(round(construction_tokens.total_tokens * (turns_seen_total / turns_total)))
                    if (turns_total > 0 and turns_seen_total > 0)
                    else construction_tokens.total_tokens
                ),
            },
            "end_to_end": {
                "prompt_tokens_total": qa_tokens.prompt_tokens + construction_tokens.prompt_tokens,
                "completion_tokens_total": qa_tokens.completion_tokens + construction_tokens.completion_tokens,
                "total_tokens_total": qa_tokens.total_tokens + construction_tokens.total_tokens,
            },
        },
        "time": {
            "qa": {
                "avg_seconds_per_q": _mean(qa_durations),
                "total_seconds": float(sum(qa_durations)),
            },
            "construction": {
                "avg_seconds_per_scene": _mean(construction_durations),
                "total_seconds": float(sum(construction_durations)),
                "total_seconds_est_full": (
                    float(sum(construction_durations) * (turns_seen_total / turns_total))
                    if (turns_total > 0 and turns_seen_total > 0)
                    else float(sum(construction_durations))
                ),
            },
            "end_to_end": {
                "total_seconds": float(sum(qa_durations) + sum(construction_durations)),
            },
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3 DialSim analyzer: F1 + retrieval proxy + tokens/time breakdown.")
    parser.add_argument("--input", type=str, required=True, help="predictions.jsonl path")
    parser.add_argument("--output", type=str, required=True, help="output json path")
    args = parser.parse_args()

    res = analyze(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
