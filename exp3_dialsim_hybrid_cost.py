import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import tiktoken


@dataclass
class Pricing:
    name: str
    input_per_m: float
    output_per_m: float

    def cost(self, prompt_tokens: float, completion_tokens: float) -> float:
        return (prompt_tokens / 1_000_000.0) * self.input_per_m + (completion_tokens / 1_000_000.0) * self.output_per_m


def build_amem_keyword_prompt(question: str) -> str:
    # Must match AgenticMemory/run_dialsim_streaming_eval.py:generate_keywords_llm
    return (
        "Given the following question, generate several keywords, using 'cosmos' as the separator.\n"
        f"Question: {question}\n"
        'Format your response as a JSON object with a "keywords" field containing the selected text.\n'
        'Example response format:\n{"keywords": "keyword1, keyword2, keyword3"}'
    )


def sum_amem_keyword_prompt_tokens(predictions_jsonl: Path, enc: tiktoken.Encoding) -> int:
    total = 0
    with predictions_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            q = str(r.get("question", "") or "")
            total += len(enc.encode(build_amem_keyword_prompt(q)))
    return total


def sum_higmem_final_answer_prompt_tokens(run_logs: List[Path], enc: tiktoken.Encoding) -> Dict[str, Any]:
    total = 0
    counts: Dict[str, int] = {}
    totals_by_show: Dict[str, int] = {}
    for path in run_logs:
        # Log file pattern: ..._{show}_YYYYMMDD_HHMMSS.jsonl
        show = None
        for s in ("friends", "bigbang", "theoffice"):
            if f"_{s}_" in path.name:
                show = s
                break
        if show is None:
            show = path.stem
        cnt = 0
        s = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("step") != "llm_call":
                    continue
                d = r.get("data") or {}
                if d.get("caller_function") != "dialsim_final_answer_generation":
                    continue
                prompt = d.get("prompt") or ""
                s += len(enc.encode(prompt))
                cnt += 1
        counts[show] = cnt
        totals_by_show[show] = s
        total += s
    return {"total": total, "counts_by_show": counts, "totals_by_show": totals_by_show}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp3 DialSim: hybrid cost (retrieval=mini, answer=gpt-5) estimate.")
    parser.add_argument("--higmem_summary_json", type=str, required=True)
    parser.add_argument("--amem_summary_json", type=str, required=True)
    parser.add_argument("--amem_predictions_jsonl", type=str, required=True)
    parser.add_argument("--higmem_run_log_jsonl", type=str, nargs="+", required=True, help="Per-show run log jsonl(s).")
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    higmem = _load_json(Path(args.higmem_summary_json))
    amem = _load_json(Path(args.amem_summary_json))

    # QA totals come from API token_usage aggregations in predictions.jsonl.
    higmem_qa_prompt = int(higmem["tokens"]["qa"]["prompt_tokens_total"])
    higmem_qa_comp = int(higmem["tokens"]["qa"]["completion_tokens_total"])
    higmem_q = int(higmem["num_questions"])

    amem_qa_prompt = int(amem["tokens"]["qa"]["prompt_tokens_total"])
    amem_qa_comp = int(amem["tokens"]["qa"]["completion_tokens_total"])
    amem_q = int(amem["num_questions"])

    # A-Mem: split QA prompt into (keyword generation) + (final answer prompt)
    amem_kw_prompt_est = sum_amem_keyword_prompt_tokens(Path(args.amem_predictions_jsonl), enc)
    amem_answer_prompt_est = max(0, amem_qa_prompt - amem_kw_prompt_est)

    # HiGMem: compute final-answer prompt tokens from logged prompts (tiktoken).
    higmem_answer = sum_higmem_final_answer_prompt_tokens([Path(p) for p in args.higmem_run_log_jsonl], enc)
    higmem_answer_prompt_est = int(higmem_answer["total"])
    higmem_retrieval_prompt_est = max(0, higmem_qa_prompt - higmem_answer_prompt_est)

    # Pricing: keep identical to Exp4.
    price_mini = Pricing(name="gpt-4o-mini", input_per_m=0.075, output_per_m=0.3)
    price_gpt5 = Pricing(name="gpt-5", input_per_m=0.625, output_per_m=5.0)

    report = {
        "inputs": {
            "higmem_summary_json": args.higmem_summary_json,
            "amem_summary_json": args.amem_summary_json,
            "amem_predictions_jsonl": args.amem_predictions_jsonl,
            "higmem_run_log_jsonl": args.higmem_run_log_jsonl,
        },
        "token_split": {
            "higmem": {
                "num_questions": higmem_q,
                "qa_prompt_tokens_total": higmem_qa_prompt,
                "qa_completion_tokens_total": higmem_qa_comp,
                "answer_prompt_tokens_est": higmem_answer_prompt_est,
                "retrieval_prompt_tokens_est": higmem_retrieval_prompt_est,
                "answer_prompt_tokens_est_avg": (higmem_answer_prompt_est / higmem_q) if higmem_q else None,
                "retrieval_prompt_tokens_est_avg": (higmem_retrieval_prompt_est / higmem_q) if higmem_q else None,
                "answer_prompt_debug": higmem_answer,
            },
            "amem": {
                "num_questions": amem_q,
                "qa_prompt_tokens_total": amem_qa_prompt,
                "qa_completion_tokens_total": amem_qa_comp,
                "retrieval_keyword_prompt_tokens_est": amem_kw_prompt_est,
                "answer_prompt_tokens_est": amem_answer_prompt_est,
                "retrieval_keyword_prompt_tokens_est_avg": (amem_kw_prompt_est / amem_q) if amem_q else None,
                "answer_prompt_tokens_est_avg": (amem_answer_prompt_est / amem_q) if amem_q else None,
            },
        },
        "pricing": {
            price_mini.name: {
                "amem_qa_cost_usd": price_mini.cost(amem_qa_prompt, amem_qa_comp),
                "higmem_qa_cost_usd": price_mini.cost(higmem_qa_prompt, higmem_qa_comp),
            },
            price_gpt5.name: {
                "amem_qa_cost_usd": price_gpt5.cost(amem_qa_prompt, amem_qa_comp),
                "higmem_qa_cost_usd": price_gpt5.cost(higmem_qa_prompt, higmem_qa_comp),
            },
            "hybrid_retrieval_mini_answer_gpt5_input_only": {
                "amem_input_cost_usd_est": (amem_kw_prompt_est / 1_000_000.0) * price_mini.input_per_m
                + (amem_answer_prompt_est / 1_000_000.0) * price_gpt5.input_per_m,
                "higmem_input_cost_usd_est": (higmem_retrieval_prompt_est / 1_000_000.0) * price_mini.input_per_m
                + (higmem_answer_prompt_est / 1_000_000.0) * price_gpt5.input_per_m,
            },
        },
        "notes": [
            "This hybrid estimate matches Exp4 style: retrieval uses gpt-4o-mini, final answer uses gpt-5, and we approximate cost using INPUT tokens only.",
            "A-Mem keyword-generation prompt tokens are estimated with tiktoken and subtracted from QA prompt tokens to approximate final-answer prompt tokens.",
            "HiGMem final-answer prompt tokens are estimated with tiktoken over the logged prompts where caller_function=='dialsim_final_answer_generation'.",
            "Totals use API token_usage aggregations from predictions.jsonl (may include overhead beyond the raw user prompt text).",
        ],
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(json.dumps(report["pricing"]["hybrid_retrieval_mini_answer_gpt5_input_only"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
