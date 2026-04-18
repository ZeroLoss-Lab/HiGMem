import os
# Avoid HF network calls during long eval runs (models are already cached on this machine).
# User can override by setting HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE in the environment.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Load local .env (keeps API key/base out of command line; safe no-op if missing).
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
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
from memory_layer import LLMController
from load_dataset import load_locomo_dataset
from utils import calculate_metrics, aggregate_metrics
from fphm_core import FPHMSystem
import prompts


def build_category_prompt(category: int, context: str, question: str, qa=None) -> str:
    """
    Build the final QA prompt using the LoCoMo prompt family aligned with A-Mem.
    """
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


def generate_keyword_query(llm_controller: LLMController, original_query: str) -> dict:
    prompt = prompts.QUERY_REWRITING_PROMPT.format(original_query=original_query)
    schema = {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keyword_query": {"type": "string"},
                "profile_retrieval_keys": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["keyword_query", "profile_retrieval_keys"]
        }
    }
    try:
        response_str = llm_controller.llm.get_completion(
            prompt,
            response_format={"type": "json_schema", "json_schema": schema},
            temperature=0.0,
        )
        data = json.loads(response_str)
        return {
            "keyword_query": data.get("keyword_query", " ".join(original_query.lower().replace("?", "").split())),
            "profile_retrieval_keys": data.get("profile_retrieval_keys", [])
        }
    except Exception:
        return {
            "keyword_query": " ".join(original_query.lower().replace("?", "").split()),
            "profile_retrieval_keys": []
        }


def process_single_sample(args, sample, base_run_name, output_dir, is_single_sample_run: bool):
    if is_single_sample_run:
        log_dir = "fphm_logs"
        checkpoint_dir = "checkpoints"
        file_run_name = base_run_name
    else:
        log_dir = output_dir
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        file_run_name = f"{base_run_name}_sample_{sample.sample_id}"

    os.makedirs(checkpoint_dir, exist_ok=True)

    llm_controller = LLMController(
        backend=args.backend, model=args.model,
        api_key=args.api_key, api_base=args.api_base
    )

    final_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{file_run_name}_final.pkl")

    memory_system = FPHMSystem(
        llm_controller=llm_controller,
        run_name=file_run_name,
        log_dir=log_dir,
        use_character_profile=(not args.ablation_no_profile),
        use_event_title_mode=args.ablation_event_title_only,
        use_event_metadata_mode=args.ablation_event_metadata_only,
        use_attribute_focused_profile=args.ablation_attribute_profile,
        k_event_affiliation=args.k_event,
        ablation_no_fact_judgment=args.ablation_no_fact_judgment,
        ablation_no_filter=args.ablation_no_filter,
        ablation_no_link=args.ablation_no_link,
        ablation_no_event=args.ablation_no_event,
        ablation_mpnet_retrieval=args.ablation_mpnet_retrieval
    )

    if os.path.exists(final_checkpoint_path):
        print(f"Sample {sample.sample_id}: Resuming from final checkpoint: {final_checkpoint_path}")
        with open(final_checkpoint_path, "rb") as f:
            saved_state = pickle.load(f)
        memory_system.__dict__.update(saved_state)
    else:
        print(f"Sample {sample.sample_id}: No final checkpoint found. Building memory from scratch.")
        all_turns = []
        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]
            sorted_session_turns = sorted(session.turns, key=lambda t: int(t.dia_id.split(':')[1]))
            for turn in sorted_session_turns:
                all_turns.append((turn, session.date_time))

        if args.max_turns is not None:
            all_turns = all_turns[: max(0, int(args.max_turns))]

        for turn, date_time in tqdm(all_turns, desc=f"Sample {sample.sample_id} Turns", leave=False):
            memory_system.add_turn(
                turn_id=turn.dia_id,
                turn_content=turn.text,
                speaker=turn.speaker,
                timestamp=date_time
            )

        print(f"Sample {sample.sample_id}: Finalizing memory build process...")
        memory_system.finalize_memory_build()
        print(f"Sample {sample.sample_id}: Finalizing indices...")
        memory_system.build_indices()

        print(f"Sample {sample.sample_id}: Saving final checkpoint before QA...")
        try:
            state_to_save = memory_system.__dict__.copy()
            if 'executor' in state_to_save: del state_to_save['executor']
            if 'llm' in state_to_save: del state_to_save['llm']
            if 'logger' in state_to_save: del state_to_save['logger']
            with open(final_checkpoint_path, "wb") as f:
                pickle.dump(state_to_save, f)
            print(f"Sample {sample.sample_id}: Successfully saved final checkpoint to {final_checkpoint_path}")
        except Exception as e:
            print(f"Sample {sample.sample_id}: Error saving final checkpoint: {e}")

    print(f"Sample {sample.sample_id}: Performing QA evaluation...")
    sample_metrics, sample_categories = [], []
    qa_items = list(sample.qa)
    if args.max_questions is not None:
        qa_items = qa_items[: max(0, int(args.max_questions))]
    for qa in tqdm(qa_items, desc=f"Sample {sample.sample_id} QA", leave=False):
        qa_start_time = time.time()
        if args.disable_query_rewriting_llm:
            keyword_query = qa.question
            profile_keys = []
            memory_system.logger.log("query_rewriting_disabled", {"original": qa.question})
        elif args.ablation_mpnet_retrieval:
            keyword_query = memory_system.generate_query_llm(qa.question)
            profile_keys = []
            memory_system.logger.log("query_rewriting_mpnet", {"original": qa.question, "rewritten_query": keyword_query})
        else:
            query_data = generate_keyword_query(llm_controller, qa.question)
            keyword_query = query_data["keyword_query"]
            profile_keys = query_data["profile_retrieval_keys"]
            memory_system.logger.log("query_rewriting", {"original": qa.question, "rewritten_data": query_data})
        context = memory_system.retrieve_for_query(
            original_query=qa.question,
            keyword_query=keyword_query,
            profile_retrieval_keys=profile_keys,
            k_profile=args.k_profile,
            k_event=args.k_event,
            k_turn=args.k_turn
        )
        final_prompt = build_category_prompt(
            category=qa.category, context=context, question=qa.question, qa=qa
        )
        temperature = 0.0
        answer_json = memory_system._get_llm_json_response(
            final_prompt,
            {"name": "response",
             "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
            caller='final_answer_generation', temperature=temperature
        )
        prediction = answer_json.get("answer",
                                     "Could not generate answer.") if answer_json else "Could not generate answer."
        qa_end_time = time.time()
        qa_duration = qa_end_time - qa_start_time
        metrics = calculate_metrics(prediction, qa.final_answer)
        sample_metrics.append(metrics)
        sample_categories.append(qa.category)
        memory_system.logger.log("qa_result", {
            "question": qa.question, "final_prompt_to_llm": final_prompt, "prediction": prediction,
            "reference": qa.final_answer, "category": qa.category, "metrics": metrics,
            "duration_seconds": qa_duration
        })

    memory_system.executor.shutdown(wait=True)
    return sample_metrics, sample_categories


def run_evaluation(args):
    run_name_parts = [args.model.replace('/', '_')]
    ablation_tags = []
    if args.ablation_no_profile: ablation_tags.append('no_profile')
    if args.ablation_no_event: ablation_tags.append('no_event')
    if args.ablation_event_title_only: ablation_tags.append('event_title')
    if args.ablation_event_metadata_only: ablation_tags.append('event_meta')
    if args.ablation_attribute_profile: ablation_tags.append('attr_profile')
    if args.ablation_no_fact_judgment: ablation_tags.append('no_fact_judge')
    if args.ablation_no_filter: ablation_tags.append('no_filter')
    if args.ablation_no_link: ablation_tags.append('no_link')
    if args.ablation_mpnet_retrieval: ablation_tags.append('mpnet_retrieval')
    if not ablation_tags:
        run_name_parts.append('full_system')
    else:
        ablation_tags.sort()
        run_name_parts.extend(ablation_tags)

    if args.max_turns is not None:
        run_name_parts.append(f'maxturn{int(args.max_turns)}')
    if args.max_questions is not None:
        run_name_parts.append(f'maxq{int(args.max_questions)}')
    # Include key hyperparams in the run name to prevent accidentally resuming from an incompatible checkpoint.
    # NOTE: k_event also affects memory construction (event affiliation), so it MUST be part of the checkpoint key.
    run_name_parts.append(f'kevent{int(args.k_event)}')
    # Query rewriting is a retrieval-time knob (cost/time tradeoff). Keep it explicit in run names so
    # sample-level toggle tests don't overwrite each other.
    run_name_parts.append('noqrw' if args.disable_query_rewriting_llm else 'qrw')
    run_name_parts.append('sync')
    base_run_name = '_'.join(run_name_parts)

    print("Loading LoCoMo dataset...")
    samples = load_locomo_dataset(args.dataset)

    all_metrics, all_categories = [], []

    if args.sample_index is not None:
        if args.sample_index < len(samples):
            sample = samples[args.sample_index]
            print(f"Running in single-sample mode for sample index: {args.sample_index}")
            all_metrics, all_categories = process_single_sample(
                args, sample, base_run_name, output_dir=None, is_single_sample_run=True
            )
        else:
            print(f"Error: Sample index {args.sample_index} is out of bounds.")
            return
    else:
        print(f"Running in parallel mode for all samples with {args.num_workers} workers "
              f"({args.parallel_backend}).")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_run_dir = os.path.join("fphm_runs", f"{base_run_name}_{timestamp}")
        os.makedirs(main_run_dir, exist_ok=True)
        print(f"All outputs for this run will be saved in: {main_run_dir}")

        executor_cls = ProcessPoolExecutor if args.parallel_backend == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=args.num_workers) as executor:
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
                    sample_metrics, sample_categories = future.result()
                    all_metrics.extend(sample_metrics)
                    all_categories.extend(sample_categories)
                except Exception as e:
                    print(f"A sample process failed: {e}")

    if not all_metrics:
        print("No metrics were generated. Evaluation might have failed.")
        return

    print("\n--- Overall Evaluation Summary ---")
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    print(json.dumps(aggregate_results, indent=2))

    if args.sample_index is None:
        result_file_path = os.path.join(main_run_dir, "aggregated_results.json")
    else:
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_filename = f"results_{base_run_name}_{timestamp}.json"
        result_file_path = os.path.join(results_dir, result_filename)

    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, indent=2, ensure_ascii=False)
    print(f"Aggregated results saved to {result_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FPHM evaluation on LoCoMo dataset.")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--backend", type=str, default="openai", help="LLM backend")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="API key")
    parser.add_argument("--api_base", type=str,
                        default=os.getenv("OPENAI_API_BASE"),
                        help="API base URL (end with /v1). Defaults to env OPENAI_API_BASE.")
    parser.add_argument("--sample_index", type=int, default=None,
                        help="Run on a single sample index. Omit to run on all.")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel workers for multi-sample evaluation.")
    parser.add_argument("--parallel-backend", type=str, default="thread", choices=["thread", "process"],
                        help="Parallel backend for multi-sample evaluation. "
                             "'thread' shares one CUDA context/VRAM; 'process' isolates but duplicates VRAM.")
    parser.add_argument("--max_turns", type=int, default=None,
                        help="(Debug/pilot) Limit the number of dialogue turns used to build memory.")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="(Debug/pilot) Limit the number of QA items evaluated.")

    parser.add_argument("--ablation-no-profile", action="store_true",
                        help="Run ablation study without character profiles.")
    parser.add_argument("--ablation-event-title-only", action="store_true",
                        help="Ablation: Use event titles and fact lists instead of full summaries.")
    parser.add_argument("--ablation-event-metadata-only", action="store_true",
                        help="Ablation: Use event titles, keywords, and tags, but no fact lists for large events.")
    parser.add_argument("--ablation-attribute-profile", action="store_true",
                        help="Ablation: Use attribute-focused character profiles instead of narrative summaries.")
    parser.add_argument("--ablation-no-fact-judgment", action="store_true",
                        help="Ablation (event_title mode only): Directly extract facts without judging event relevance again.")
    parser.add_argument("--ablation-no-filter", action="store_true",
                        help="Ablation: Disable LLM-based relevance filtering/judgment in the QA retrieval phase.")
    parser.add_argument("--ablation-no-link", action="store_true",
                        help="Ablation: Disable immediate context linking during TurnNote creation.")
    parser.add_argument("--ablation-no-event", action="store_true",
                        help="Ablation: Disable both Event and Profile layers, operating only on TurnNotes.")
    parser.add_argument("--ablation-mpnet-retrieval", action="store_true",
                        help="Ablation: Use MPNet retriever and new declarative query rewriting.")
    parser.add_argument("--disable_query_rewriting_llm", action="store_true",
                         help="Disable LLM-based query rewriting/keyword generation; use the original question for retrieval.")

    parser.add_argument("--k_profile", type=int, default=3, help="Top-K for character profile retrieval.")
    # Paper default is k_event=10 (previously hard-wired to 7 by mistake).
    parser.add_argument("--k_event", type=int, default=10, help="Top-K for event summary retrieval.")
    parser.add_argument("--k_turn", type=int, default=10, help="Top-K for turn note retrieval.")

    args = parser.parse_args()
    run_evaluation(args)
