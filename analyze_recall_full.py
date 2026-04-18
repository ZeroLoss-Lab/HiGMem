# analyze_recall_full.py
import json
import argparse
from collections import defaultdict
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import pickle
import re
import glob
import tiktoken
from typing import List, Dict, Any, Tuple

from load_dataset import load_locomo_dataset
from fphm_core import FPHMSystem


def build_evidence_to_event_map(fphm_system: FPHMSystem) -> dict:
    mapping = defaultdict(list)
    if not hasattr(fphm_system, 'events') or not fphm_system.events:
        return {}

    for event_id, event in fphm_system.events.items():
        for turn_id in event.turn_note_ids:
            if event_id not in mapping[turn_id]:
                mapping[turn_id].append(event_id)
    return dict(mapping)


def calculate_token_cost_from_llm_call(log_entry: dict, enc: tiktoken.Encoding) -> Tuple[str, int]:
    step = log_entry.get("step")
    if step != "llm_call":
        return "unknown", 0

    data = log_entry.get("data", {})
    prompt_text = data.get("prompt", "")
    response_text = data.get("raw_response", "")
    caller = data.get("caller_function", "unknown_caller")

    CONSTRUCTION_CALLERS = {
        '_create_and_link_turn_note', '_create_turn_note_isolated', '_decide_event_affiliation',
        '_update_event', '_update_event_default', '_update_event_adaptive_small',
        '_update_event_adaptive_large', '_update_event_extract_facts_only',
        '_update_event_metadata_small', '_update_event_metadata_large',
        'profile_update_decision', 'final_profile_update_decision',
        '_update_profile', '_update_profile_attribute_focused', '_update_profile_narrative_summary'
    }
    QA_CALLERS = {
        'generate_keyword_query', 'generate_query_llm', 'final_answer_generation',
        '_judge_relevance_parallel_turn', '_judge_relevance_parallel_event', '_judge_relevance_parallel_profile',
        '_judge_relevance_sequential_turn', '_judge_relevance_sequential_event', '_judge_relevance_sequential_profile',
        'predict_events_from_profile', 'predict_turns_from_event'
    }

    cost_category = "unknown"
    clean_caller = caller.lstrip('_')
    if clean_caller in CONSTRUCTION_CALLERS or caller in CONSTRUCTION_CALLERS:
        cost_category = f"construction_{caller.lstrip('_')}"
    elif clean_caller in QA_CALLERS or caller in QA_CALLERS:
        cost_category = f"qa_{caller.lstrip('_')}"

    if prompt_text or response_text:
        response_text = response_text or ""
        total_tokens = len(enc.encode(prompt_text)) + len(enc.encode(response_text))
        return cost_category, total_tokens

    return "unknown", 0


def analyze_construction_cost(logs: list, enc: tiktoken.Encoding) -> dict:
    total_construction_costs = defaultdict(int)
    total_construction_time = 0.0

    for entry in logs:
        step = entry.get("step")
        if step == "llm_call":
            cost_category, token_count = calculate_token_cost_from_llm_call(entry, enc)
            if token_count > 0 and cost_category.startswith("construction_"):
                total_construction_costs[cost_category] += token_count
        elif step == "timing_add_turn":
            total_construction_time += entry.get("data", {}).get("duration_seconds", 0.0)

    total_construction_costs['total_token_construction'] = sum(
        v for k, v in total_construction_costs.items() if k.startswith('construction_'))
    total_construction_costs['total_time_construction_seconds'] = total_construction_time
    return dict(total_construction_costs)


def analyze_qa_log(logs: list, qa_evidence_map: dict, evidence_to_event_map: dict, run_name: str,
                   enc: tiktoken.Encoding) -> pd.DataFrame:
    recall_data = []
    logs_by_question = defaultdict(list)
    current_question = None
    in_qa_phase = False

    for log in logs:
        data = log.get("data", {})
        step = log.get("step")
        question_in_log = data.get("question") or data.get("original_query") or data.get("query")
        if question_in_log:
            if not in_qa_phase: in_qa_phase = True
            current_question = question_in_log
        if in_qa_phase and current_question:
            logs_by_question[current_question].append(log)

    for question, entries in logs_by_question.items():
        if question not in qa_evidence_map: continue
        gold_evidence_set = qa_evidence_map[question]
        if not gold_evidence_set: continue

        gold_event_set = {eid for ev_id in gold_evidence_set if ev_id in evidence_to_event_map for eid in
                          evidence_to_event_map[ev_id]}

        analysis = {"question": question, "run_name": run_name, "gold_evidence_count": len(gold_evidence_set),
                    "gold_event_count": len(gold_event_set)}
        recalled_turns_vector, fused_turns_before_expansion, final_context_turns, selected_event_ids = set(), set(), set(), set()
        token_costs = defaultdict(int)
        time_cost_qa = 0.0

        for entry in entries:
            step = entry.get("step")
            data = entry.get("data", {})

            if step == "llm_call":
                cost_category, token_count = calculate_token_cost_from_llm_call(entry, enc)
                if token_count > 0 and cost_category.startswith("qa_"):
                    agg_category = cost_category
                    if 'relevance' in cost_category:
                        agg_category = 'qa_relevance_judgment'
                    elif 'predict_events' in cost_category:
                        agg_category = 'qa_predict_events_from_profile'
                    elif 'predict_turns' in cost_category:
                        agg_category = 'qa_predict_turns_from_event'
                    elif 'generate_keyword' in cost_category or 'generate_query' in cost_category:
                        agg_category = 'qa_query_rewriting'
                    elif 'final_answer' in cost_category:
                        agg_category = 'qa_final_answer_generation'
                    token_costs[agg_category] += token_count
            elif step == "qa_result":
                time_cost_qa += data.get("duration_seconds", 0.0)
            elif step == "initial_recall":
                recalled_turns_vector.update(data.get("recalled_turns", []))
            elif step in ["judge_relevance_event", "judge_relevance_sequential_event"]:
                selected_event_ids.update(data.get("selected", []))
            elif step == "predict_turns_from_event":
                fused_turns_before_expansion.update(data.get("predicted_turns", []))
                fused_turns_before_expansion.update(recalled_turns_vector)
            elif step in ["judge_relevance_turn", "judge_relevance_sequential_turn"]:
                final_context_turns.update(data.get("selected", []))

        hits_initial_turns = gold_evidence_set.intersection(recalled_turns_vector)
        analysis['recall_initial_turns_vector'] = len(hits_initial_turns) / len(
            gold_evidence_set) if gold_evidence_set else 0
        hits_after_event_fusion = gold_evidence_set.intersection(fused_turns_before_expansion)
        analysis['recall_after_event_fusion'] = len(hits_after_event_fusion) / len(
            gold_evidence_set) if gold_evidence_set else 0
        hits_final_turns = gold_evidence_set.intersection(final_context_turns)
        analysis['recall_final_turns'] = len(hits_final_turns) / len(gold_evidence_set) if gold_evidence_set else 0
        lost_in_reranking = hits_after_event_fusion - hits_final_turns
        analysis['lost_in_reranking_count'] = len(lost_in_reranking)
        analysis['num_turns_final_selected'] = len(final_context_turns)
        analysis['precision_final_turns'] = len(hits_final_turns) / len(
            final_context_turns) if final_context_turns else 0
        analysis.update(token_costs)
        analysis['total_token_qa_cost'] = sum(v for k, v in token_costs.items() if k.startswith('qa_'))
        analysis['total_time_qa_seconds'] = time_cost_qa
        recall_data.append(analysis)
    return pd.DataFrame(recall_data)


def analyze_sample_data(log_path: Path, checkpoint_path: Path, qa_maps: dict, run_name: str,
                        enc: tiktoken.Encoding) -> Tuple[pd.DataFrame, dict]:
    if not checkpoint_path.exists():
        print(f"\n\033[93mWarning: Checkpoint not found: {checkpoint_path}. Skipping sample.\033[0m")
        return pd.DataFrame(), {}
    with open(checkpoint_path, "rb") as f:
        saved_state = pickle.load(f)
    temp_system = FPHMSystem(llm_controller=None, run_name="temp_for_analysis")
    temp_log_path = Path(temp_system.logger.log_file)
    try:
        temp_system.__dict__.update(saved_state)
        evidence_to_event_map = build_evidence_to_event_map(temp_system)
        all_logs = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                try:
                    all_logs.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"\n\033[91mERROR: Skipping malformed JSON line {i + 1} in {log_path.name}\033[0m")
                    continue
        construction_costs = analyze_construction_cost(all_logs, enc)
        raw_df = analyze_qa_log(all_logs, qa_maps['evidence'], evidence_to_event_map, run_name, enc)
        if not raw_df.empty:
            raw_df['category'] = raw_df['question'].map(qa_maps['category'])
        return raw_df, construction_costs
    finally:
        if temp_log_path.exists(): os.remove(temp_log_path)


def aggregate_run_results(raw_dfs: List[pd.DataFrame], construction_costs_list: List[dict],
                          config_name: str) -> pd.DataFrame:
    if not raw_dfs: return pd.DataFrame()
    full_raw_df = pd.concat(raw_dfs, ignore_index=True)
    if full_raw_df.empty: return pd.DataFrame()

    qa_agg_metrics = {
        'num_questions': ('question', 'count'), 'recall_initial_turns_vector': ('recall_initial_turns_vector', 'mean'),
        'recall_after_event_fusion': ('recall_after_event_fusion', 'mean'),
        'recall_final_turns': ('recall_final_turns', 'mean'),
        'precision_final_turns': ('precision_final_turns', 'mean'),
        'avg_turns_final_selected': ('num_turns_final_selected', 'mean'),
        'total_lost_in_reranking': ('lost_in_reranking_count', 'sum'),
        'avg_token_qa_total': ('total_token_qa_cost', 'mean'),
        'avg_time_qa_seconds': ('total_time_qa_seconds', 'mean'),
        'avg_token_qa_relevance': ('qa_relevance_judgment', 'mean'),
        'avg_token_qa_query_rewrite': ('qa_query_rewriting', 'mean'),
        'avg_token_qa_predict_events': ('qa_predict_events_from_profile', 'mean'),
        'avg_token_qa_predict_turns': ('qa_predict_turns_from_event', 'mean'),
        'avg_token_qa_final_answer': ('qa_final_answer_generation', 'mean'),
    }
    existing_agg_metrics = {n: (s, f) for n, (s, f) in qa_agg_metrics.items() if
                            s in full_raw_df.columns or n == 'num_questions'}
    summary_by_category = full_raw_df.groupby('category').agg(**existing_agg_metrics).reset_index()

    simple_agg_spec = {s: (lambda x, f=f: getattr(x, f)(skipna=True)) for n, (s, f) in existing_agg_metrics.items() if
                       n != 'num_questions'}
    rename_map = {s: n for n, (s, f) in existing_agg_metrics.items() if n != 'num_questions'}
    overall_series = full_raw_df.agg(simple_agg_spec).rename(rename_map)
    summary_overall = overall_series.to_frame().T
    summary_overall['category'] = 'Overall'
    summary_overall['num_questions'] = len(full_raw_df)
    final_summary = pd.concat([summary_by_category, summary_overall], ignore_index=True)

    total_construction_costs = defaultdict(float)
    for costs in construction_costs_list:
        for key, value in costs.items():
            total_construction_costs[key] += value
    for key, value in total_construction_costs.items():
        final_summary.loc[final_summary['category'] == 'Overall', key] = value

    final_summary['config_name'] = config_name
    return final_summary


def analyze_f1_scores(config_name: str, result_file_path: Path) -> dict:
    if not result_file_path or not result_file_path.exists():
        return {"config_name": config_name, "Overall_F1": "No Result File"}
    with open(result_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    f1_scores = {"config_name": config_name}
    if 'overall' in data and 'f1' in data['overall']:
        f1_scores['Overall_F1'] = data['overall']['f1'].get('mean', 0)
    for i in range(1, 6):
        cat_key = f"category_{i}"
        if cat_key in data and 'f1' in data[cat_key]:
            f1_scores[f'Cat_{i}_F1'] = data[cat_key]['f1'].get('mean', 0)
    return f1_scores


def main():
    parser = argparse.ArgumentParser(description="Analyze recall and F1 performance for FPHM runs.")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to the LoCoMo dataset file.")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save output CSVs.")
    parser.add_argument("--logs_dir", type=str, default="fphm_logs", help="Directory for single-sample logs.")
    parser.add_argument("--runs_dir", type=str, default="fphm_runs", help="Directory for full-dataset run outputs.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for single-sample results_...json files.")
    args = parser.parse_args()
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Could not initialize tiktoken, token analysis will be skipped. Error: {e}")
        enc = None

    print("Loading dataset to build QA-Evidence map...")
    samples = load_locomo_dataset(Path(args.dataset))
    qa_maps = {
        'evidence': {qa.question: set(qa.evidence) for s in samples for qa in s.qa},
        'category': {qa.question: qa.category for s in samples for qa in s.qa}
    }
    print(f"Built map for {len(qa_maps['evidence'])} questions.")

    configurations = defaultdict(list)

    runs_dir = Path(args.runs_dir)
    full_run_pattern = re.compile(r"(.+)_(\d{8}_\d{6})")
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir(): continue
            match = full_run_pattern.match(run_dir.name)
            if not match: continue

            config_name, timestamp = match.groups()

            log_paths, checkpoint_paths = [], []
            is_complete = True
            for sample_dir in run_dir.glob("sample_*"):
                found_logs = list(sample_dir.glob("*.jsonl"))
                found_checkpoints = list((sample_dir / "checkpoints").glob("*_final.pkl"))

                if len(found_logs) == 1 and len(found_checkpoints) == 1:
                    log_paths.append(found_logs[0])
                    checkpoint_paths.append(found_checkpoints[0])
                else:
                    is_complete = False
                    break

            result_path = run_dir / "aggregated_results.json"

            if is_complete and result_path.exists():
                configurations[config_name].append({
                    "timestamp": timestamp, "type": "full_dataset", "log_paths": log_paths,
                    "checkpoint_paths": checkpoint_paths, "result_path": result_path
                })

    single_run_log_pattern = re.compile(r"run_(.+)_\d{8}_\d{6}\.jsonl")
    for log_file in Path(args.logs_dir).glob("*.jsonl"):
        match = single_run_log_pattern.match(log_file.name)
        if not match: continue

        full_name_from_log = match.group(1)
        if "sample_" in full_name_from_log: continue

        try:
            timestamp = '_'.join(log_file.stem.split('_')[-2:])
            config_name = log_file.stem.replace(f"run_{timestamp}", "")[4:]  # remove "run_" and timestamp
        except IndexError:
            continue

        checkpoint_path = Path("checkpoints") / f"checkpoint_{config_name}_final.pkl"
        result_files = list(Path(args.results_dir).glob(f"results_{config_name}_*.json"))

        if checkpoint_path.exists() and result_files:
            latest_result = max(result_files, key=os.path.getmtime)
            configurations[config_name].append({
                "timestamp": timestamp, "type": "single_sample", "log_paths": [log_file],
                "checkpoint_paths": [checkpoint_path], "result_path": latest_result
            })

    final_runs_to_analyze = []
    for config_name, run_list in configurations.items():
        if not run_list: continue
        latest_run = sorted(run_list, key=lambda x: x['timestamp'], reverse=True)[0]
        latest_run['config_name'] = config_name
        final_runs_to_analyze.append(latest_run)

    if not final_runs_to_analyze:
        print(
            "\n\033[91mError: No valid and complete experiment runs found. Check file paths and naming conventions.\033[0m")
        return

    print(f"\nDiscovered and selected the latest run for {len(final_runs_to_analyze)} unique configurations:")
    for run in sorted(final_runs_to_analyze, key=lambda x: x['config_name']):
        print(f"  - {run['config_name']} (Type: {run['type']}, Timestamp: {run['timestamp']})")

    all_recall_summaries, all_raw_dfs, all_f1_data = [], [], []

    print("\n--- Running Analysis on Latest Runs ---")
    for run in tqdm(final_runs_to_analyze, desc="Processing Configurations"):
        config_name = run['config_name']
        sample_raw_dfs, sample_construction_costs = [], []

        if enc:
            for log_p, chk_p in zip(run['log_paths'], run['checkpoint_paths']):
                raw_df, costs = analyze_sample_data(log_p, chk_p, qa_maps, config_name, enc)
                if not raw_df.empty: sample_raw_dfs.append(raw_df)
                if costs: sample_construction_costs.append(costs)

        if sample_raw_dfs:
            summary_df = aggregate_run_results(sample_raw_dfs, sample_construction_costs, config_name)
            all_recall_summaries.append(summary_df)
            all_raw_dfs.extend(sample_raw_dfs)

        f1_summary = analyze_f1_scores(config_name, run['result_path'])
        all_f1_data.append(f1_summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if all_recall_summaries:
        final_recall_df = pd.concat(all_recall_summaries, ignore_index=True).round(4)
        cols_order = [
            'config_name', 'category', 'num_questions', 'recall_initial_turns_vector', 'recall_after_event_fusion',
            'recall_final_turns', 'precision_final_turns', 'avg_turns_final_selected', 'total_lost_in_reranking',
            'avg_token_qa_total', 'avg_time_qa_seconds', 'total_token_construction', 'total_time_construction_seconds'
        ]
        ordered_cols = [c for c in cols_order if c in final_recall_df.columns]
        remaining_cols = sorted([c for c in final_recall_df.columns if c not in ordered_cols])
        final_recall_df = final_recall_df[ordered_cols + remaining_cols]

        output_path = output_dir / "recall_and_cost_summary_BEST_49.csv"
        final_recall_df.to_csv(output_path, index=False)
        print("\n\n--- Aggregated Recall & Cost Summary (Overall) ---")
        print(final_recall_df[final_recall_df['category'] == 'Overall'].fillna('NA').to_string())
        print(f"\nFull recall & cost summary saved to {output_path}")

    if all_raw_dfs:
        pd.concat(all_raw_dfs, ignore_index=True).to_csv(output_dir / "raw_analysis_per_question.csv", index=False)
        print(f"Raw per-question analysis saved to {output_dir / 'raw_analysis_per_question.csv'}")

    if all_f1_data:
        f1_df = pd.DataFrame(all_f1_data).set_index('config_name').fillna('N/A')
        for col in f1_df.columns:
            if col != 'config_name': f1_df[col] = f1_df[col].apply(
                lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else x)
        print("\n\n--- F1 Score Summary ---")
        print(f1_df.to_string())
        f1_df.to_csv(output_dir / "f1_summary.csv")
        print(f"\nF1 summary saved to {output_dir / 'f1_summary.csv'}")


if __name__ == "__main__":
    main()
