# full_context_test.py
import os

try:
    from dotenv import load_dotenv  # type: ignore

    _ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(_ENV_PATH):
        load_dotenv(_ENV_PATH)
except Exception:
    pass

import argparse
import json
import random
from tqdm import tqdm
from collections import defaultdict
import sys
from datetime import datetime
import time
import tiktoken
from concurrent.futures import ProcessPoolExecutor, as_completed

from memory_layer import LLMController
from load_dataset import load_locomo_dataset, LoCoMoSample, Turn
from utils import calculate_metrics, aggregate_metrics
from fphm_logger import FPHMLogger


def get_all_turns_map(sample: LoCoMoSample) -> dict[str, Turn]:
    all_turns_map = {}
    for session in sample.conversation.sessions.values():
        for turn in session.turns:
            setattr(turn, 'timestamp', session.date_time)
            all_turns_map[turn.dia_id] = turn
    return all_turns_map


def build_context_from_turns(turn_objects: list[Turn]) -> str:
    final_context_parts = []
    try:
        sorted_turns = sorted(turn_objects, key=lambda t: (getattr(t, 'timestamp', ''), t.dia_id))
    except (TypeError, ValueError):
        sorted_turns = sorted(turn_objects, key=lambda t: t.dia_id)

    for t in sorted_turns:
        turn_string = (
            f"talk start time:{getattr(t, 'timestamp', 'N/A')}"
            f"memory content: Speaker {t.speaker}says : {t.text}"
            f"memory context: "
            f"memory keywords: []"
            f"memory tags: []"
        )
        final_context_parts.append(turn_string)

    return "\n".join(final_context_parts)


def build_category_prompt(category: int, context: str, question: str, qa=None) -> str:
    prompt = ""
    if category == 5:
        trap_answer = qa.adversarial_answer
        answer_tmp = []
        if random.random() < 0.5:
            answer_tmp.append('Not mentioned in the conversation')
            answer_tmp.append(trap_answer)
        else:
            answer_tmp.append(trap_answer)
            answer_tmp.append('Not mentioned in the conversation')
        prompt = f"""
                        Based on the context: {context}, answer the following question. {question} 
                        Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
                        """
    elif category == 2:
        prompt = f"""
                        Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
                        Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   
                        Question: {question} Short answer:
                        """
    elif category == 3:
        prompt = f"""
                        Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
                        Question: {question} Short answer:
                        """
    else:
        prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
                            Question: {question} Short answer:
                            """
    return prompt.strip()


def analyze_and_print_top_bottom_results(qa_results: list, output_dir: str, run_name: str):
    by_category = defaultdict(list)
    for result in qa_results:
        by_category[result['category']].append(result)

    analysis_str_builder = []
    analysis_str_builder.append("=" * 80)
    analysis_str_builder.append(" " * 25 + "DETAILED QA PERFORMANCE ANALYSIS")
    analysis_str_builder.append("=" * 80 + "\n")

    for category_id in sorted(by_category.keys()):
        category_results = by_category[category_id]
        analysis_str_builder.append(f"\n\n{'=' * 20} ANALYSIS FOR CATEGORY {category_id} {'=' * 20}\n")

        category_results.sort(key=lambda x: x['f1'], reverse=True)

        analysis_str_builder.append("\n--- Top 5 BEST Answers (by F1 Score) ---\n")
        for res in category_results[:5]:
            analysis_str_builder.append(f"F1: {res['f1']:.4f}\n")
            analysis_str_builder.append(f"  Q: {res['question']}\n")
            analysis_str_builder.append(f"  A (Prediction): {res['prediction']}\n")
            analysis_str_builder.append(f"  A (Reference):  {res['reference']}\n")
            evidence_text = res.get('evidence_context', 'N/A')
            truncated_evidence = evidence_text[:1000] + '...' if len(evidence_text) > 1000 else evidence_text
            analysis_str_builder.append(f"  Evidence Seen by Model (truncated):\n---\n{truncated_evidence}\n---\n")

        analysis_str_builder.append("\n--- Top 5 WORST Answers (by F1 Score) ---\n")
        for res in category_results[-5:][::-1]:
            analysis_str_builder.append(f"F1: {res['f1']:.4f}\n")
            analysis_str_builder.append(f"  Q: {res['question']}\n")
            analysis_str_builder.append(f"  A (Prediction): {res['prediction']}\n")
            analysis_str_builder.append(f"  A (Reference):  {res['reference']}\n")
            evidence_text = res.get('evidence_context', 'N/A')
            truncated_evidence = evidence_text[:1000] + '...' if len(evidence_text) > 1000 else evidence_text
            analysis_str_builder.append(f"  Evidence Seen by Model (truncated):\n---\n{truncated_evidence}\n---\n")

    final_analysis_str = "".join(analysis_str_builder)
    print(final_analysis_str)

    analysis_file_path = os.path.join(output_dir, f"analysis_top_bottom_{run_name}.txt")
    try:
        with open(analysis_file_path, 'w', encoding='utf-8') as f:
            f.write(final_analysis_str)
        print(f"\nDetailed QA analysis saved to {analysis_file_path}")
    except Exception as e:
        print(f"\nError saving analysis file: {e}")


def process_single_sample(args, sample, base_run_name, output_dir, is_single_sample_run: bool):
    if is_single_sample_run:
        log_dir = "fphm_logs"
        file_run_name = base_run_name
    else:
        log_dir = output_dir
        file_run_name = f"{base_run_name}_sample_{sample.sample_id}"

    llm_controller = LLMController(
        backend=args.backend, model=args.model,
        api_key=args.api_key, api_base=args.api_base
    )
    logger = FPHMLogger(log_dir=log_dir, run_name=file_run_name)

    sample_metrics, sample_categories = [], []
    qa_results_for_analysis = []

    all_qa_durations = []
    all_prompt_tokens = []
    all_completion_tokens = []
    all_total_tokens = []
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Warning: Could not initialize tiktoken for token estimation. Token counts will be zero. Error: {e}")
        encoding = None

    all_turns_map = get_all_turns_map(sample)

    print(f"\nRunning QA evaluation for sample {sample.sample_id} with full context...")
    full_context_turn_objects = list(all_turns_map.values())
    context = build_context_from_turns(full_context_turn_objects)

    for qa in tqdm(sample.qa, desc=f"Sample {sample.sample_id} QA", leave=False):
        if not qa.evidence:
            continue

        temperature = 0.0
        final_prompt = build_category_prompt(
            category=qa.category,
            context=context,
            question=qa.question,
            qa=qa
        )

        start_time = time.time()
        try:
            response = llm_controller.llm.get_completion(
                final_prompt,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "response",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                               "required": ["answer"]}
                }},
                temperature=temperature
            )
            answer_json = json.loads(response)
            prediction = answer_json.get("answer", "Could not generate answer.")
        except Exception as e:
            prediction = f"Error during LLM call: {e}"
            print(f"Warning: LLM call failed for question '{qa.question}'. Error: {e}")
        end_time = time.time()
        duration = end_time - start_time
        all_qa_durations.append(duration)

        if encoding:
            prompt_tokens = len(encoding.encode(final_prompt))
            completion_tokens = len(encoding.encode(prediction))
        else:
            prompt_tokens, completion_tokens = 0, 0
        total_tokens = prompt_tokens + completion_tokens

        all_prompt_tokens.append(prompt_tokens)
        all_completion_tokens.append(completion_tokens)
        all_total_tokens.append(total_tokens)

        metrics = calculate_metrics(prediction, qa.final_answer)
        sample_metrics.append(metrics)
        sample_categories.append(qa.category)

        qa_results_for_analysis.append({
            "question": qa.question,
            "prediction": prediction,
            "reference": qa.final_answer,
            "category": qa.category,
            "f1": metrics.get('f1', 0.0),
            "evidence_context": context,
            "prompt_used": final_prompt
        })

        logger.log("qa_result", {
            "mode": "full_context",
            "question": qa.question,
            "context_turn_ids": list(all_turns_map.keys()),
            "final_prompt_to_llm": final_prompt,
            "prediction": prediction,
            "reference": qa.final_answer,
            "category": qa.category,
            "metrics": metrics,
            "duration_seconds": duration,
            "prompt_tokens_estimated": prompt_tokens,
            "completion_tokens_estimated": completion_tokens,
            "total_tokens_estimated": total_tokens
        })

    performance_metrics = {}
    if all_qa_durations:
        performance_metrics = {
            "total_questions_processed": len(all_qa_durations),
            "average_duration_seconds_per_qa": sum(all_qa_durations) / len(all_qa_durations),
            "average_prompt_tokens_estimated_per_qa": sum(all_prompt_tokens) / len(all_prompt_tokens),
            "average_completion_tokens_estimated_per_qa": sum(all_completion_tokens) / len(all_completion_tokens),
            "average_total_tokens_estimated_per_qa": sum(all_total_tokens) / len(all_total_tokens),
        }

    return sample_metrics, sample_categories, qa_results_for_analysis, performance_metrics


def run_full_context_test(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = f"full_context_test_{args.model.replace('/', '_')}"

    print("Loading LoCoMo dataset...")
    samples = load_locomo_dataset(args.dataset)

    all_metrics, all_categories = [], []
    all_qa_results_for_analysis = []
    all_performance_metrics = []

    if args.sample_index is not None:
        if args.sample_index < len(samples):
            sample = samples[args.sample_index]
            print(f"Running in single-sample mode for sample index: {args.sample_index}")
            metrics, categories, qa_results, perf_metrics = process_single_sample(
                args, sample, base_run_name, output_dir=None, is_single_sample_run=True
            )
            all_metrics.extend(metrics)
            all_categories.extend(categories)
            all_qa_results_for_analysis.extend(qa_results)
            if perf_metrics:
                all_performance_metrics.append(perf_metrics)
        else:
            print(f"Error: Sample index {args.sample_index} is out of bounds.")
            return
    else:
        print(f"Running in parallel mode for all {len(samples)} samples with {args.num_workers} workers.")
        main_run_dir = os.path.join("fphm_runs", f"{base_run_name}_{timestamp}")
        os.makedirs(main_run_dir, exist_ok=True)
        print(f"All outputs for this run will be saved in: {main_run_dir}")

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for sample in samples:
                sample_output_dir = os.path.join(main_run_dir, f"sample_{sample.sample_id}")
                os.makedirs(sample_output_dir, exist_ok=True)
                future = executor.submit(
                    process_single_sample, args, sample, base_run_name, sample_output_dir, is_single_sample_run=False
                )
                futures.append(future)

            progress_bar = tqdm(as_completed(futures), total=len(samples), desc="Processing Samples")
            for future in progress_bar:
                try:
                    sample_metrics, sample_categories, qa_results, perf_metrics = future.result()
                    all_metrics.extend(sample_metrics)
                    all_categories.extend(sample_categories)
                    all_qa_results_for_analysis.extend(qa_results)
                    if perf_metrics:
                        all_performance_metrics.append(perf_metrics)
                except Exception as e:
                    print(f"A sample process failed: {e}")

    if not all_metrics:
        print("No metrics were generated. Evaluation might have failed.")
        return

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    if all_performance_metrics:
        total_qas = sum(p['total_questions_processed'] for p in all_performance_metrics)
        if total_qas > 0:
            avg_duration = sum(p['average_duration_seconds_per_qa'] * p['total_questions_processed'] for p in all_performance_metrics) / total_qas
            avg_prompt_tokens = sum(p['average_prompt_tokens_estimated_per_qa'] * p['total_questions_processed'] for p in all_performance_metrics) / total_qas
            avg_completion_tokens = sum(p['average_completion_tokens_estimated_per_qa'] * p['total_questions_processed'] for p in all_performance_metrics) / total_qas
            avg_total_tokens = sum(p['average_total_tokens_estimated_per_qa'] * p['total_questions_processed'] for p in all_performance_metrics) / total_qas

            overall_performance = {
                "total_questions_processed": total_qas,
                "average_duration_seconds_per_qa": avg_duration,
                "average_prompt_tokens_estimated_per_qa": avg_prompt_tokens,
                "average_completion_tokens_estimated_per_qa": avg_completion_tokens,
                "average_total_tokens_estimated_per_qa": avg_total_tokens,
            }
            aggregate_results["performance_metrics"] = overall_performance

    print(f"\n--- Evaluation Summary for Mode: Full Context ---")
    print(json.dumps(aggregate_results, indent=2))

    if args.sample_index is None:
        results_dir = main_run_dir
    else:
        results_dir = f"full_context_test_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)

    final_run_name = f"{base_run_name}_{timestamp}"
    result_file = os.path.join(results_dir, f"results_{final_run_name}.json")
    with open(result_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"Results saved to {result_file}")

    detailed_results_file = os.path.join(results_dir, f"detailed_qa_results_{final_run_name}.json")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_results_for_analysis, f, indent=2, ensure_ascii=False)
    print(f"Detailed QA results saved to {detailed_results_file}")

    if all_qa_results_for_analysis:
        analyze_and_print_top_bottom_results(all_qa_results_for_analysis, results_dir, final_run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full-context baseline testing on LoCoMo dataset.")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to the dataset file")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--backend", type=str, help="LLM backend")
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--api_base", type=str, help="API base URL")
    parser.add_argument("--sample_index", type=int, help="Run on a single sample index. Omit to run on all.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel processes for multi-sample evaluation.")

    if len(sys.argv) == 1:
        print("No command-line arguments provided. Running with default quick-test settings...")
        args = argparse.Namespace(
            dataset="data/locomo10.json",
            model="gpt-4o-mini",
            backend="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            sample_index=0,
            num_workers=4,
        )
    else:
        print("Command-line arguments detected. Parsing as usual...")
        parser.set_defaults(
            model="gpt-4o-mini",
            backend="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            sample_index=None,
        )
        args = parser.parse_args()

    run_full_context_test(args)
