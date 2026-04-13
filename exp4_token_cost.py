import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import tiktoken


@dataclass
class Pricing:
    name: str
    input_per_m: float
    output_per_m: float

    def cost(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_per_m + (completion_tokens / 1_000_000.0) * self.output_per_m


def build_amem_keyword_prompt(question: str) -> str:
    # Must match AgenticMemory/test_advanced.py:advancedMemAgent.generate_query_llm
    return f"""Given the following question, generate several keywords, using 'cosmos' as the separator.
                Question: {question}
                Format your response as a JSON object with a "keywords" field containing the selected text. 
                Example response format:
                {{"keywords": "keyword1, keyword2, keyword3"}}"""


def sum_amem_keyword_prompt_tokens(dataset_path: Path, enc: tiktoken.Encoding) -> int:
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    total = 0
    for sample in data:
        for qa in sample.get("qa", []):
            q = qa.get("question", "")
            total += len(enc.encode(build_amem_keyword_prompt(q)))
    return total


def load_higmem_overall_row(summary_csv: Path, config_name: str) -> Dict[str, Any]:
    with open(summary_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("config_name") == config_name and row.get("category") == "Overall":
                return row
    raise FileNotFoundError(f"Overall row not found for config_name={config_name} in {summary_csv}")


def _to_int(x: Optional[str]) -> int:
    if x is None:
        return 0
    x = str(x).strip()
    if not x:
        return 0
    return int(float(x))


def _to_float(x: Optional[str]) -> float:
    if x is None:
        return 0.0
    x = str(x).strip()
    if not x:
        return 0.0
    return float(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp4: End-to-End Token Cost (retrieval + generation) analysis.")
    parser.add_argument(
        "--amem_results_json",
        type=str,
        required=True,
        help="Path to AgenticMemory results JSON (contains API token usage totals).",
    )
    parser.add_argument(
        "--amem_dataset",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "AgenticMemory" / "data" / "locomo10.json"),
        help="Path to locomo10.json used by A-Mem (for keyword prompt token estimation).",
    )
    parser.add_argument(
        "--higmem_summary_csv",
        type=str,
        default=str(Path(__file__).parent / "analysis_results" / "recall_and_cost_summary_BEST_49.csv"),
        help="HiGMem analysis CSV (from analyze_recall.py).",
    )
    parser.add_argument(
        "--higmem_config_name",
        type=str,
        default="gpt-4o-mini_event_meta_no_profile_kevent10_qrw_sync",
        help="Config name row to read from the HiGMem summary CSV.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(Path(__file__).parent / "analysis_results" / "exp4_token_cost_report.json"),
        help="Where to write the JSON report.",
    )
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    amem_results_path = Path(args.amem_results_json)
    amem_dataset_path = Path(args.amem_dataset)
    higmem_summary_path = Path(args.higmem_summary_csv)

    amem = json.loads(amem_results_path.read_text(encoding="utf-8"))
    amem_total_q = int(amem.get("total_questions", 0))
    amem_qa_stats = amem.get("performance_stats", {}).get("qa", {})
    amem_qa_prompt = int(amem_qa_stats.get("prompt_tokens", 0))
    amem_qa_comp = int(amem_qa_stats.get("completion_tokens", 0))

    amem_mc_stats = amem.get("performance_stats", {}).get("memory_construction", {}) or {}
    amem_mc_prompt = int(amem_mc_stats.get("prompt_tokens", 0))
    amem_mc_comp = int(amem_mc_stats.get("completion_tokens", 0))
    amem_mc_total = int(amem_mc_stats.get("total_tokens", amem_mc_prompt + amem_mc_comp))
    amem_mc_time = float(amem_mc_stats.get("duration_seconds", 0.0) or 0.0)

    amem_kw_prompt_est = sum_amem_keyword_prompt_tokens(amem_dataset_path, enc)
    amem_answer_prompt_est = max(0, amem_qa_prompt - amem_kw_prompt_est)

    higmem_row = load_higmem_overall_row(higmem_summary_path, args.higmem_config_name)
    higmem_n = _to_int(higmem_row.get("num_questions"))
    higmem_prompt_total = _to_int(higmem_row.get("total_prompt_token_qa_total"))
    higmem_prompt_answer = _to_int(higmem_row.get("total_prompt_tokens_final_answer"))
    higmem_token_total = _to_int(higmem_row.get("total_token_qa_total"))
    higmem_comp_total = max(0, higmem_token_total - higmem_prompt_total)
    higmem_prompt_retrieval = max(0, higmem_prompt_total - higmem_prompt_answer)

    higmem_construction_total = _to_int(higmem_row.get("total_token_construction"))
    higmem_construction_time = _to_float(higmem_row.get("total_time_construction_seconds"))
    higmem_e2e_total = _to_int(higmem_row.get("total_token_end_to_end"))
    higmem_e2e_time = _to_float(higmem_row.get("total_time_end_to_end_seconds"))

    price_mini = Pricing(name="gpt-4o-mini", input_per_m=0.075, output_per_m=0.3)
    price_gpt5 = Pricing(name="gpt-5", input_per_m=0.625, output_per_m=5.0)

    report = {
        "amem": {
            "total_questions": amem_total_q,
            "qa_prompt_tokens_total": amem_qa_prompt,
            "qa_completion_tokens_total": amem_qa_comp,
            "retrieval_keyword_prompt_tokens_est": amem_kw_prompt_est,
            "answer_prompt_tokens_est": amem_answer_prompt_est,
            "avg_prompt_tokens_per_question": (amem_qa_prompt / amem_total_q) if amem_total_q else None,
            "avg_answer_prompt_tokens_est_per_question": (amem_answer_prompt_est / amem_total_q) if amem_total_q else None,
            "memory_construction": {
                "duration_seconds": amem_mc_time,
                "prompt_tokens_total": amem_mc_prompt,
                "completion_tokens_total": amem_mc_comp,
                "total_tokens_total": amem_mc_total,
                "avg_total_tokens_per_question_amortized": (amem_mc_total / amem_total_q) if amem_total_q else None,
            },
            "end_to_end": {
                "duration_seconds": amem_mc_time + float(amem.get("performance_stats", {}).get("qa", {}).get("duration_seconds", 0.0) or 0.0),
                "prompt_tokens_total": amem_qa_prompt + amem_mc_prompt,
                "completion_tokens_total": amem_qa_comp + amem_mc_comp,
                "total_tokens_total": (amem_qa_prompt + amem_qa_comp) + amem_mc_total,
            },
        },
        "higmem": {
            "config_name": args.higmem_config_name,
            "num_questions_in_summary": higmem_n,
            "qa_prompt_tokens_total": higmem_prompt_total,
            "qa_completion_tokens_total_est": higmem_comp_total,
            "retrieval_prompt_tokens_total_est": higmem_prompt_retrieval,
            "answer_prompt_tokens_total_est": higmem_prompt_answer,
            "avg_prompt_tokens_per_question": (higmem_prompt_total / higmem_n) if higmem_n else None,
            "avg_answer_prompt_tokens_per_question": (higmem_prompt_answer / higmem_n) if higmem_n else None,
            "memory_construction": {
                "duration_seconds": higmem_construction_time,
                "total_tokens_total": higmem_construction_total,
                "avg_total_tokens_per_question_amortized": (higmem_construction_total / amem_total_q) if amem_total_q else None,
            },
            "end_to_end": {
                "duration_seconds": higmem_e2e_time,
                "total_tokens_total": higmem_e2e_total,
            },
        },
        "pricing": {
            price_mini.name: {
                "amem_qa_cost_usd": price_mini.cost(amem_qa_prompt, amem_qa_comp),
                "higmem_qa_cost_usd_est": price_mini.cost(higmem_prompt_total, higmem_comp_total),
            },
            price_gpt5.name: {
                "amem_qa_cost_usd": price_gpt5.cost(amem_qa_prompt, amem_qa_comp),
                "higmem_qa_cost_usd_est": price_gpt5.cost(higmem_prompt_total, higmem_comp_total),
            },
            "hybrid_retrieval_mini_answer_gpt5": {
                # Input-only approximation; output tokens are small for these short answers.
                "amem_input_cost_usd_est": (amem_kw_prompt_est / 1_000_000.0) * price_mini.input_per_m
                + (amem_answer_prompt_est / 1_000_000.0) * price_gpt5.input_per_m,
                "higmem_input_cost_usd_est": (higmem_prompt_retrieval / 1_000_000.0) * price_mini.input_per_m
                + (higmem_prompt_answer / 1_000_000.0) * price_gpt5.input_per_m,
            },
        },
        "notes": [
            "A-Mem uses API-reported tokens from results JSON; retrieval keyword prompt tokens are estimated via tiktoken and subtracted to approximate answer prompt tokens.",
            "HiGMem tokens come from tiktoken over logged prompts/responses (see analyze_recall.py); some older logs contain malformed JSON lines which may drop a few QAs from the summary.",
            "This report counts user-prompt text only (system messages are excluded), matching the existing HiGMem analysis convention.",
        ],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps(report["pricing"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
