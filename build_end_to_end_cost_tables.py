import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import tiktoken


@dataclass
class Pricing:
    name: str
    input_per_m: float
    output_per_m: float

    def cost_usd(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_per_m + (completion_tokens / 1_000_000.0) * self.output_per_m


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_row(path: Path, *, config_name: str, category: str = "Overall") -> Dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("config_name") == config_name and row.get("category") == category:
                return row
    raise FileNotFoundError(f"Row not found: config_name={config_name}, category={category}, csv={path}")


def _to_int(x: Optional[str]) -> int:
    if x is None:
        return 0
    s = str(x).strip()
    if not s:
        return 0
    return int(float(s))


def _to_float(x: Optional[str]) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if not s:
        return 0.0
    return float(s)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Keep analysis robust against occasional malformed log lines.
                continue


def build_amem_keyword_prompt_locomo(question: str) -> str:
    # Must match AgenticMemory/test_advanced.py:advancedMemAgent.generate_query_llm
    # NOTE: keep the indentation/spaces exactly, otherwise token estimation drifts.
    return f"""Given the following question, generate several keywords, using 'cosmos' as the separator.
                Question: {question}
                Format your response as a JSON object with a "keywords" field containing the selected text. 
                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""


def build_amem_keyword_prompt_dialsim(question: str) -> str:
    # Must match AgenticMemory/run_dialsim_streaming_eval.py:generate_keywords_llm
    return (
        "Given the following question, generate several keywords, using 'cosmos' as the separator.\n"
        f"Question: {question}\n"
        'Format your response as a JSON object with a "keywords" field containing the selected text.\n'
        'Example response format:\n{"keywords": "keyword1, keyword2, keyword3"}'
    )


def sum_amem_keyword_prompt_tokens_from_locomo_dataset(dataset_json: Path, enc: tiktoken.Encoding) -> int:
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    total = 0
    for sample in data:
        for qa in sample.get("qa", []):
            q = str(qa.get("question", "") or "")
            total += len(enc.encode(build_amem_keyword_prompt_locomo(q)))
    return total


def sum_amem_keyword_completion_tokens_from_locomo_traces(qa_traces_dir: Path, enc: tiktoken.Encoding) -> int:
    # Approximate: model output is a JSON object {"keywords": "..."} (as instructed by the prompt).
    total = 0
    for p in sorted(qa_traces_dir.glob("qa_trace_sample_*.jsonl")):
        for r in _iter_jsonl(p):
            keywords = (((r.get("retrieval_trace") or {}).get("keywords")) or "")
            resp = json.dumps({"keywords": str(keywords)}, ensure_ascii=False)
            total += len(enc.encode(resp))
    return total


def sum_amem_keyword_prompt_tokens_from_dialsim_predictions(predictions_jsonl: Path, enc: tiktoken.Encoding) -> int:
    total = 0
    for r in _iter_jsonl(predictions_jsonl):
        q = str(r.get("question", "") or "")
        total += len(enc.encode(build_amem_keyword_prompt_dialsim(q)))
    return total


def sum_amem_keyword_completion_tokens_from_dialsim_predictions(predictions_jsonl: Path, enc: tiktoken.Encoding) -> int:
    # Approximate: wrap the saved retrieval_query into the JSON format the prompt asks for.
    total = 0
    for r in _iter_jsonl(predictions_jsonl):
        keywords = str(r.get("retrieval_query", "") or "")
        resp = json.dumps({"keywords": keywords}, ensure_ascii=False)
        total += len(enc.encode(resp))
    return total


def load_amem_avgk_from_recall_summary(recall_fixed_k_summary_csv: Path) -> float:
    with recall_fixed_k_summary_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("method") == "A-Mem" and row.get("turns") == "full":
                return float(row.get("avg_turns_actual") or 0.0)
    raise FileNotFoundError(f"Could not find A-Mem full row in {recall_fixed_k_summary_csv}")


# Keep in sync with H-Mem/analyze_recall.py.
CONSTRUCTION_CALLERS = {
    "_create_and_link_turn_note",
    "_create_turn_note_isolated",
    "_decide_event_affiliation",
    "_update_event",
    "_update_event_default",
    "_update_event_adaptive_small",
    "_update_event_adaptive_large",
    "_update_event_extract_facts_only",
    "_update_event_metadata_small",
    "_update_event_metadata_large",
    "profile_update_decision",
    "final_profile_update_decision",
    "_update_profile",
    "_update_profile_attribute_focused",
    "_update_profile_narrative_summary",
}


def sum_higmem_locomo_construction_prompt_completion(run_dir: Path, *, config_name: str, enc: tiktoken.Encoding) -> Dict[str, Any]:
    """Compute construction prompt/completion tokens directly from the full-dataset run logs.

    We only need this because the aggregated locomo10 summary CSV stores construction *total* tokens
    but not prompt/completion breakdown.
    """
    prompt_total = 0
    completion_total = 0
    llm_calls = 0

    log_paths = sorted(run_dir.glob("sample_*/*.jsonl"))
    if not log_paths:
        raise FileNotFoundError(f"No jsonl logs under {run_dir}")

    for p in log_paths:
        # Safety: ensure we only read logs belonging to this config.
        if f"run_{config_name}_" not in p.name:
            continue
        for e in _iter_jsonl(p):
            if e.get("step") != "llm_call":
                continue
            d = e.get("data") or {}
            caller = str(d.get("caller_function") or "")
            clean = caller.lstrip("_")
            if (caller not in CONSTRUCTION_CALLERS) and (clean not in CONSTRUCTION_CALLERS):
                continue
            prompt = str(d.get("prompt") or "")
            resp = str(d.get("raw_response") or "")
            prompt_total += len(enc.encode(prompt)) if prompt else 0
            completion_total += len(enc.encode(resp)) if resp else 0
            llm_calls += 1

    return {
        "prompt_tokens_total": int(prompt_total),
        "completion_tokens_total": int(completion_total),
        "total_tokens_total": int(prompt_total + completion_total),
        "llm_calls_counted": int(llm_calls),
    }


def sum_higmem_dialsim_answer_prompt_completion(run_logs: List[Path], enc: tiktoken.Encoding) -> Dict[str, Any]:
    prompt_total = 0
    completion_total = 0
    calls = 0
    for p in run_logs:
        for e in _iter_jsonl(p):
            if e.get("step") != "llm_call":
                continue
            d = e.get("data") or {}
            if d.get("caller_function") != "dialsim_final_answer_generation":
                continue
            prompt = str(d.get("prompt") or "")
            resp = str(d.get("raw_response") or "")
            prompt_total += len(enc.encode(prompt)) if prompt else 0
            completion_total += len(enc.encode(resp)) if resp else 0
            calls += 1
    return {
        "prompt_tokens_total": int(prompt_total),
        "completion_tokens_total": int(completion_total),
        "total_tokens_total": int(prompt_total + completion_total),
        "llm_calls_counted": int(calls),
    }


def fmt_m(x: float) -> str:
    return f"{x/1_000_000.0:.3f}M"


def fmt_h(seconds: float) -> str:
    return f"{seconds/3600.0:.2f}h"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build end-to-end cost tables (construction + retrieval + big-model answering).")
    parser.add_argument("--output_json", type=str, default=str(Path(__file__).parent / "analysis_results" / "end_to_end_cost_tables.json"))
    parser.add_argument("--output_md", type=str, default=str(Path(__file__).parent / "analysis_results" / "end_to_end_cost_tables.md"))
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    price_mini = Pricing(name="gpt-4o-mini", input_per_m=0.075, output_per_m=0.3)
    price_gpt5 = Pricing(name="gpt-5", input_per_m=0.625, output_per_m=5.0)

    # --- locomo10 paths ---
    locomo_higmem_config = "gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync"
    locomo_higmem_run_dir = Path("H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync_20260222_003900")
    locomo_higmem_summary_csv = Path("H-Mem/analysis_results/recall_and_cost_summary_BEST_49.csv")
    locomo_higmem_agg_json = locomo_higmem_run_dir / "aggregated_results.json"

    locomo_amem_results_json = Path("AgenticMemory/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408.json")
    locomo_amem_dataset_json = Path("AgenticMemory/data/locomo10.json")
    locomo_amem_traces_dir = Path("AgenticMemory/qa_traces/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408")
    locomo_amem_recall_csv = locomo_amem_traces_dir / "recall_fixed_k_summary.csv"

    # --- DialSim paths ---
    dialsim_higmem_summary_json = Path("H-Mem/analysis_results/exp3_dialsim_higmem_kevent10_noqrw_summary.json")
    dialsim_amem_summary_json = Path("H-Mem/analysis_results/exp3_dialsim_amem_stream_k50_summary.json")
    dialsim_amem_predictions_jsonl = Path("AgenticMemory/dialsim_runs/gpt-4o-mini_openai_amem_dialsim_stream_20260222_001841/predictions.jsonl")
    dialsim_higmem_run_logs = [
        Path("H-Mem/dialsim_runs/gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_20260222_001838/run_gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_friends_20260222_001838.jsonl"),
        Path("H-Mem/dialsim_runs/gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_20260222_001838/run_gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_bigbang_20260222_001838.jsonl"),
        Path("H-Mem/dialsim_runs/gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_20260222_001838/run_gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_theoffice_20260222_001838.jsonl"),
    ]

    # ---------------- locomo10: HiGMem ----------------
    locomo_higmem_summary_row = _read_csv_row(locomo_higmem_summary_csv, config_name=locomo_higmem_config, category="Overall")
    locomo_higmem_f1 = float((_read_json(locomo_higmem_agg_json).get("overall", {}).get("f1", {}) or {}).get("mean", 0.0) or 0.0)
    locomo_higmem_avgk = _to_float(locomo_higmem_summary_row.get("avg_turns_final_selected"))

    # QA (note: summary excludes 4 evidence-empty questions; n=1982)
    locomo_higmem_n = _to_int(locomo_higmem_summary_row.get("num_questions"))
    locomo_higmem_qa_prompt = _to_int(locomo_higmem_summary_row.get("total_prompt_token_qa_total"))
    locomo_higmem_qa_total = _to_int(locomo_higmem_summary_row.get("total_token_qa_total"))
    locomo_higmem_qa_comp = max(0, locomo_higmem_qa_total - locomo_higmem_qa_prompt)
    locomo_higmem_qa_time_s = _to_float(locomo_higmem_summary_row.get("total_time_qa_seconds"))

    locomo_higmem_answer_prompt = _to_int(locomo_higmem_summary_row.get("total_prompt_tokens_final_answer"))
    # total answer tokens for the final-answer LLM call only; estimate from avg * n (matches analyze_recall).
    avg_answer_total = _to_float(locomo_higmem_summary_row.get("avg_token_qa_final_answer"))
    locomo_higmem_answer_total_est = int(round(avg_answer_total * locomo_higmem_n))
    locomo_higmem_answer_comp_est = max(0, locomo_higmem_answer_total_est - locomo_higmem_answer_prompt)
    locomo_higmem_retr_prompt = max(0, locomo_higmem_qa_prompt - locomo_higmem_answer_prompt)
    locomo_higmem_retr_comp = max(0, locomo_higmem_qa_comp - locomo_higmem_answer_comp_est)

    # Construction (prompt/comp from logs; time from summary)
    locomo_higmem_mc_from_logs = sum_higmem_locomo_construction_prompt_completion(
        locomo_higmem_run_dir, config_name=locomo_higmem_config, enc=enc
    )
    locomo_higmem_mc_time_s = _to_float(locomo_higmem_summary_row.get("total_time_construction_seconds"))
    # Sanity: summary stores total tokens only
    locomo_higmem_mc_total_summary = int(float(locomo_higmem_summary_row.get("total_token_construction") or 0.0))

    # ---------------- locomo10: A-Mem ----------------
    locomo_amem = _read_json(locomo_amem_results_json)
    locomo_amem_f1 = float((locomo_amem.get("aggregate_metrics", {}).get("overall", {}).get("f1", {}) or {}).get("mean", 0.0) or 0.0)
    locomo_amem_avgk = float(load_amem_avgk_from_recall_summary(locomo_amem_recall_csv))

    amem_mc = locomo_amem.get("performance_stats", {}).get("memory_construction", {}) or {}
    amem_qa = locomo_amem.get("performance_stats", {}).get("qa", {}) or {}
    locomo_amem_mc_prompt = int(amem_mc.get("prompt_tokens", 0))
    locomo_amem_mc_comp = int(amem_mc.get("completion_tokens", 0))
    locomo_amem_mc_time_s = float(amem_mc.get("duration_seconds", 0.0) or 0.0)

    locomo_amem_qa_prompt = int(amem_qa.get("prompt_tokens", 0))
    locomo_amem_qa_comp = int(amem_qa.get("completion_tokens", 0))
    locomo_amem_qa_time_s = float(amem_qa.get("duration_seconds", 0.0) or 0.0)

    locomo_amem_kw_prompt_est = sum_amem_keyword_prompt_tokens_from_locomo_dataset(locomo_amem_dataset_json, enc)
    locomo_amem_kw_comp_est = sum_amem_keyword_completion_tokens_from_locomo_traces(locomo_amem_traces_dir, enc)
    locomo_amem_answer_prompt_est = max(0, locomo_amem_qa_prompt - locomo_amem_kw_prompt_est)
    locomo_amem_answer_comp_est = max(0, locomo_amem_qa_comp - locomo_amem_kw_comp_est)

    # ---------------- DialSim: summaries ----------------
    dialsim_higmem = _read_json(dialsim_higmem_summary_json)
    dialsim_amem = _read_json(dialsim_amem_summary_json)

    dialsim_higmem_f1 = float(dialsim_higmem.get("metrics", {}).get("f1_mean", 0.0) or 0.0)
    dialsim_amem_f1 = float(dialsim_amem.get("metrics", {}).get("f1_mean", 0.0) or 0.0)

    dialsim_higmem_avgk = float(dialsim_higmem.get("retrieval", {}).get("avg_k", 0.0) or 0.0)
    dialsim_amem_avgk = float(dialsim_amem.get("retrieval", {}).get("avg_k", 0.0) or 0.0)

    dialsim_higmem_mc_prompt = int(dialsim_higmem["tokens"]["construction"]["prompt_tokens_total"])
    dialsim_higmem_mc_comp = int(dialsim_higmem["tokens"]["construction"]["completion_tokens_total"])
    dialsim_higmem_mc_time_s = float(dialsim_higmem["time"]["construction"]["total_seconds"])

    dialsim_higmem_qa_prompt = int(dialsim_higmem["tokens"]["qa"]["prompt_tokens_total"])
    dialsim_higmem_qa_comp = int(dialsim_higmem["tokens"]["qa"]["completion_tokens_total"])
    dialsim_higmem_qa_time_s = float(dialsim_higmem["time"]["qa"]["total_seconds"])

    # DialSim HiGMem: answer prompt/comp from logs (exact), retrieval from subtraction.
    dialsim_higmem_answer = sum_higmem_dialsim_answer_prompt_completion(dialsim_higmem_run_logs, enc)
    dialsim_higmem_answer_prompt = int(dialsim_higmem_answer["prompt_tokens_total"])
    dialsim_higmem_answer_comp = int(dialsim_higmem_answer["completion_tokens_total"])
    dialsim_higmem_retr_prompt = max(0, dialsim_higmem_qa_prompt - dialsim_higmem_answer_prompt)
    dialsim_higmem_retr_comp = max(0, dialsim_higmem_qa_comp - dialsim_higmem_answer_comp)

    # DialSim A-Mem
    dialsim_amem_mc_prompt = int(dialsim_amem["tokens"]["construction"]["prompt_tokens_total"])
    dialsim_amem_mc_comp = int(dialsim_amem["tokens"]["construction"]["completion_tokens_total"])
    dialsim_amem_mc_time_s = float(dialsim_amem["time"]["construction"]["total_seconds"])

    dialsim_amem_qa_prompt = int(dialsim_amem["tokens"]["qa"]["prompt_tokens_total"])
    dialsim_amem_qa_comp = int(dialsim_amem["tokens"]["qa"]["completion_tokens_total"])
    dialsim_amem_qa_time_s = float(dialsim_amem["time"]["qa"]["total_seconds"])

    dialsim_amem_kw_prompt_est = sum_amem_keyword_prompt_tokens_from_dialsim_predictions(dialsim_amem_predictions_jsonl, enc)
    dialsim_amem_kw_comp_est = sum_amem_keyword_completion_tokens_from_dialsim_predictions(dialsim_amem_predictions_jsonl, enc)
    dialsim_amem_answer_prompt_est = max(0, dialsim_amem_qa_prompt - dialsim_amem_kw_prompt_est)
    dialsim_amem_answer_comp_est = max(0, dialsim_amem_qa_comp - dialsim_amem_kw_comp_est)

    def hybrid_cost(construction_prompt: int, construction_comp: int, retrieval_prompt: int, retrieval_comp: int,
                    answer_prompt: int, answer_comp: int) -> Dict[str, Any]:
        small_prompt = construction_prompt + retrieval_prompt
        small_comp = construction_comp + retrieval_comp
        big_prompt = answer_prompt
        big_comp = answer_comp
        small_cost = price_mini.cost_usd(small_prompt, small_comp)
        big_cost = price_gpt5.cost_usd(big_prompt, big_comp)
        return {
            "small_model": {
                "model": price_mini.name,
                "prompt_tokens": int(small_prompt),
                "completion_tokens": int(small_comp),
                "cost_usd": small_cost,
            },
            "big_model": {
                "model": price_gpt5.name,
                "prompt_tokens": int(big_prompt),
                "completion_tokens": int(big_comp),
                "cost_usd": big_cost,
            },
            "total_cost_usd": small_cost + big_cost,
        }

    tables: Dict[str, Any] = {
        "pricing": {
            "gpt-4o-mini": {"input_per_m": price_mini.input_per_m, "output_per_m": price_mini.output_per_m},
            "gpt-5": {"input_per_m": price_gpt5.input_per_m, "output_per_m": price_gpt5.output_per_m},
        },
        "locomo10": {
            "higmem": {
                "f1": locomo_higmem_f1,
                "avg_k": locomo_higmem_avgk,
                "memory_construction": {
                    **locomo_higmem_mc_from_logs,
                    "time_seconds": locomo_higmem_mc_time_s,
                    "total_tokens_total_summary_csv": locomo_higmem_mc_total_summary,
                },
                "qa": {
                    "num_questions_in_summary": locomo_higmem_n,
                    "prompt_tokens_total": locomo_higmem_qa_prompt,
                    "completion_tokens_total_est": locomo_higmem_qa_comp,
                    "time_seconds": locomo_higmem_qa_time_s,
                    "retrieval": {
                        "prompt_tokens_total": locomo_higmem_retr_prompt,
                        "completion_tokens_total_est": locomo_higmem_retr_comp,
                    },
                    "answer": {
                        "prompt_tokens_total": locomo_higmem_answer_prompt,
                        "completion_tokens_total_est": locomo_higmem_answer_comp_est,
                    },
                },
                "hybrid_cost_with_construction": hybrid_cost(
                    locomo_higmem_mc_from_logs["prompt_tokens_total"],
                    locomo_higmem_mc_from_logs["completion_tokens_total"],
                    locomo_higmem_retr_prompt,
                    locomo_higmem_retr_comp,
                    locomo_higmem_answer_prompt,
                    locomo_higmem_answer_comp_est,
                ),
            },
            "amem": {
                "f1": locomo_amem_f1,
                "avg_k": locomo_amem_avgk,
                "memory_construction": {
                    "prompt_tokens_total": locomo_amem_mc_prompt,
                    "completion_tokens_total": locomo_amem_mc_comp,
                    "total_tokens_total": int(locomo_amem_mc_prompt + locomo_amem_mc_comp),
                    "time_seconds": locomo_amem_mc_time_s,
                },
                "qa": {
                    "num_questions": int(locomo_amem.get("total_questions", 0) or 0),
                    "prompt_tokens_total": locomo_amem_qa_prompt,
                    "completion_tokens_total": locomo_amem_qa_comp,
                    "time_seconds": locomo_amem_qa_time_s,
                    "retrieval": {
                        "prompt_tokens_total_est": locomo_amem_kw_prompt_est,
                        "completion_tokens_total_est": locomo_amem_kw_comp_est,
                    },
                    "answer": {
                        "prompt_tokens_total_est": locomo_amem_answer_prompt_est,
                        "completion_tokens_total_est": locomo_amem_answer_comp_est,
                    },
                },
                "hybrid_cost_with_construction": hybrid_cost(
                    locomo_amem_mc_prompt,
                    locomo_amem_mc_comp,
                    locomo_amem_kw_prompt_est,
                    locomo_amem_kw_comp_est,
                    locomo_amem_answer_prompt_est,
                    locomo_amem_answer_comp_est,
                ),
            },
        },
        "dialsim_v1_1_stream_t7000_q3000": {
            "higmem": {
                "f1": dialsim_higmem_f1,
                "avg_k": dialsim_higmem_avgk,
                "memory_construction": {
                    "prompt_tokens_total": dialsim_higmem_mc_prompt,
                    "completion_tokens_total": dialsim_higmem_mc_comp,
                    "total_tokens_total": int(dialsim_higmem_mc_prompt + dialsim_higmem_mc_comp),
                    "time_seconds": dialsim_higmem_mc_time_s,
                },
                "qa": {
                    "num_questions": int(dialsim_higmem.get("num_questions", 0) or 0),
                    "prompt_tokens_total": dialsim_higmem_qa_prompt,
                    "completion_tokens_total": dialsim_higmem_qa_comp,
                    "time_seconds": dialsim_higmem_qa_time_s,
                    "retrieval": {
                        "prompt_tokens_total": dialsim_higmem_retr_prompt,
                        "completion_tokens_total": dialsim_higmem_retr_comp,
                    },
                    "answer": {
                        "prompt_tokens_total": dialsim_higmem_answer_prompt,
                        "completion_tokens_total": dialsim_higmem_answer_comp,
                        "debug": dialsim_higmem_answer,
                    },
                },
                "hybrid_cost_with_construction": hybrid_cost(
                    dialsim_higmem_mc_prompt,
                    dialsim_higmem_mc_comp,
                    dialsim_higmem_retr_prompt,
                    dialsim_higmem_retr_comp,
                    dialsim_higmem_answer_prompt,
                    dialsim_higmem_answer_comp,
                ),
            },
            "amem": {
                "f1": dialsim_amem_f1,
                "avg_k": dialsim_amem_avgk,
                "memory_construction": {
                    "prompt_tokens_total": dialsim_amem_mc_prompt,
                    "completion_tokens_total": dialsim_amem_mc_comp,
                    "total_tokens_total": int(dialsim_amem_mc_prompt + dialsim_amem_mc_comp),
                    "time_seconds": dialsim_amem_mc_time_s,
                },
                "qa": {
                    "num_questions": int(dialsim_amem.get("num_questions", 0) or 0),
                    "prompt_tokens_total": dialsim_amem_qa_prompt,
                    "completion_tokens_total": dialsim_amem_qa_comp,
                    "time_seconds": dialsim_amem_qa_time_s,
                    "retrieval": {
                        "prompt_tokens_total_est": dialsim_amem_kw_prompt_est,
                        "completion_tokens_total_est": dialsim_amem_kw_comp_est,
                    },
                    "answer": {
                        "prompt_tokens_total_est": dialsim_amem_answer_prompt_est,
                        "completion_tokens_total_est": dialsim_amem_answer_comp_est,
                    },
                },
                "hybrid_cost_with_construction": hybrid_cost(
                    dialsim_amem_mc_prompt,
                    dialsim_amem_mc_comp,
                    dialsim_amem_kw_prompt_est,
                    dialsim_amem_kw_comp_est,
                    dialsim_amem_answer_prompt_est,
                    dialsim_amem_answer_comp_est,
                ),
            },
        },
        "notes": [
            "Hybrid cost definition: construction + retrieval use gpt-4o-mini pricing; final answer uses gpt-5 pricing.",
            "All token counts are based on: A-Mem API token_usage totals; HiGMem tiktoken over logged prompt/raw_response. Splits for A-Mem keyword tokens are estimated via tiktoken.",
            "locomo10 HiGMem QA totals follow existing analysis convention (exclude 4 evidence-empty questions; n=1982).",
            "Keyword completion tokens are approximated by wrapping saved keywords into a JSON object {\"keywords\": \"...\"}, matching the prompt requirement.",
        ],
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- Markdown tables ----------------
    lines: List[str] = []
    lines.append("# End-to-End Cost Tables (Construction + Retrieval + Big-Model Answer)")
    lines.append("")
    lines.append("Pricing:")
    lines.append(f"- gpt-4o-mini: input ${price_mini.input_per_m}/M, output ${price_mini.output_per_m}/M")
    lines.append(f"- gpt-5: input ${price_gpt5.input_per_m}/M, output ${price_gpt5.output_per_m}/M")
    lines.append("")

    # Table 1: F1 / avgK / construction+QA prompt/time
    lines.append("## Table 1: F1 / avgK / Prompt Tokens + Time (Construction vs QA)")
    lines.append("")
    lines.append("| dataset | method | F1 | avgK | MC prompt | MC time | QA prompt | QA time |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    def row_table1(dataset: str, method: str, f1: float, avgk: float,
                   mc_prompt: int, mc_time_s: float, qa_prompt: int, qa_time_s: float) -> str:
        return (
            f"| {dataset} | {method} | {f1:.4f} | {avgk:.2f} | {fmt_m(mc_prompt)} | {fmt_h(mc_time_s)} | "
            f"{fmt_m(qa_prompt)} | {fmt_h(qa_time_s)} |"
        )

    # locomo10
    lines.append(
        row_table1(
            "locomo10",
            "HiGMem",
            locomo_higmem_f1,
            locomo_higmem_avgk,
            locomo_higmem_mc_from_logs["prompt_tokens_total"],
            locomo_higmem_mc_time_s,
            locomo_higmem_qa_prompt,
            locomo_higmem_qa_time_s,
        )
    )
    lines.append(
        row_table1(
            "locomo10",
            "A-Mem",
            locomo_amem_f1,
            locomo_amem_avgk,
            locomo_amem_mc_prompt,
            locomo_amem_mc_time_s,
            locomo_amem_qa_prompt,
            locomo_amem_qa_time_s,
        )
    )

    # DialSim
    lines.append(
        row_table1(
            "DialSim(v1.1,10k-sim)",
            "HiGMem",
            dialsim_higmem_f1,
            dialsim_higmem_avgk,
            dialsim_higmem_mc_prompt,
            dialsim_higmem_mc_time_s,
            dialsim_higmem_qa_prompt,
            dialsim_higmem_qa_time_s,
        )
    )
    lines.append(
        row_table1(
            "DialSim(v1.1,10k-sim)",
            "A-Mem",
            dialsim_amem_f1,
            dialsim_amem_avgk,
            dialsim_amem_mc_prompt,
            dialsim_amem_mc_time_s,
            dialsim_amem_qa_prompt,
            dialsim_amem_qa_time_s,
        )
    )

    lines.append("")
    lines.append("## Table 2: Hybrid End-to-End USD Cost (mini for construction+retrieval, gpt-5 for answer)")
    lines.append("")
    lines.append("| dataset | method | mini tokens (MC+retr) | gpt-5 tokens (answer) | mini cost | gpt-5 cost | total |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    def row_table2(dataset: str, method: str, cost: Dict[str, Any]) -> str:
        sp = int(cost["small_model"]["prompt_tokens"])
        sc = int(cost["small_model"]["completion_tokens"])
        bp = int(cost["big_model"]["prompt_tokens"])
        bc = int(cost["big_model"]["completion_tokens"])
        small_tokens = sp + sc
        big_tokens = bp + bc
        return (
            f"| {dataset} | {method} | {fmt_m(small_tokens)} | {fmt_m(big_tokens)} | "
            f"${cost['small_model']['cost_usd']:.2f} | ${cost['big_model']['cost_usd']:.2f} | ${cost['total_cost_usd']:.2f} |"
        )

    lines.append(row_table2("locomo10", "HiGMem", tables["locomo10"]["higmem"]["hybrid_cost_with_construction"]))
    lines.append(row_table2("locomo10", "A-Mem", tables["locomo10"]["amem"]["hybrid_cost_with_construction"]))
    lines.append(row_table2("DialSim(v1.1,10k-sim)", "HiGMem", tables["dialsim_v1_1_stream_t7000_q3000"]["higmem"]["hybrid_cost_with_construction"]))
    lines.append(row_table2("DialSim(v1.1,10k-sim)", "A-Mem", tables["dialsim_v1_1_stream_t7000_q3000"]["amem"]["hybrid_cost_with_construction"]))
    lines.append("")
    lines.append("Notes:")
    for n in tables["notes"]:
        lines.append(f"- {n}")
    lines.append("")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")

    # Print a quick sanity line for locomo10 construction total tokens.
    diff = locomo_higmem_mc_from_logs["total_tokens_total"] - locomo_higmem_mc_total_summary
    print(f"[sanity] locomo10 HiGMem construction_total(logs)={locomo_higmem_mc_from_logs['total_tokens_total']} vs summary_csv={locomo_higmem_mc_total_summary} (diff={diff})")


if __name__ == "__main__":
    main()
