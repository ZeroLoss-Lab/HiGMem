# --- fphm_code_changes_start ---
# oracle_test.py
import argparse
import json
import os
import random
from tqdm import tqdm
from collections import defaultdict
import sys
import time
# --- MODIFICATION START ---
# 代码内注释：导入 ProcessPoolExecutor 以支持多进程并行处理，as_completed 用于优雅地处理返回结果。
from concurrent.futures import ProcessPoolExecutor, as_completed
# --- MODIFICATION END ---

try:
    import tiktoken
except ImportError:
    print("Warning: tiktoken library not found. Token usage will not be estimated.")
    print("Please install it using: pip install tiktoken")
    tiktoken = None
from datetime import datetime

from memory_layer import LLMController
from load_dataset import load_locomo_dataset, LoCoMoSample, Turn
from utils import calculate_metrics, aggregate_metrics
from fphm_logger import FPHMLogger


def _estimate_tokens(text: str, model_name: str = "gpt-4") -> int:
    if not tiktoken:
        return 0
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


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
            f" memory content: Speaker {t.speaker} says : {t.text}"
            f" memory context: "
            f" memory keywords: []"
            f" memory tags: []"
        )
        final_context_parts.append(turn_string)

    return "\n".join(final_context_parts)


def build_category_prompt_original_oracle(category: int, context: str, question: str, qa=None) -> str:
    if category == 1:
        prompt = f"""Context:
{context}
Question: {question}
RULES (MUST FOLLOW):
1. Answer with 1-5 words ONLY - no complete sentences
2. Use Arabic numerals for numbers (e.g., "3" not "three")
3. Match the question's focus: 
   - "What kind of X" → answer the TYPE/CATEGORY (e.g., "science fiction" not plot points)
   - "What items" → list the items directly (e.g., "pens, notebooks")
   - "Where" → location only (e.g., "Paris")
4. Use generic terms when appropriate (e.g., "car" not "red sports car")
5. No extra words like "mountains" when "images of mountains" suffices
Short answer:"""
    elif category == 2:
        prompt = f"""Context:
{context}
Question: {question}
QUESTION TYPE DETECTION (Critical):
- "When did X happen?" → Answer with SPECIFIC DATE (e.g., "15 June 1957")
- "How long ago was/did X..." → Answer with RELATIVE DURATION (e.g., "10 years ago")
- "How long has X been/had..." → Answer with DURATION LENGTH (e.g., "6 months", "7 years")
CALCULATION RULES:
- "yesterday" in context → timestamp_date - 1 day
- "last year" in context → If asking "when": give the year; If asking "how long": give "X years"
- "X years" mentioned + "how long" question → Just answer "X years"
Answer format must match question type above.
Answer:"""
    elif category == 3:
        prompt = f"""Context:
{context}
Question: {question}
DETECT QUESTION TYPE AND ANSWER ACCORDINGLY:
TYPE A - Single-word trait/leaning questions:
Pattern: "What would X's [political leaning/personality/preference] be?"
→ Answer with ONE WORD only. No explanation.
Examples: "Liberal" | "Introverted" | "Artistic"
TYPE B - Yes/No likelihood questions:
Pattern: "Would X likely/probably...?"
→ Format: "Likely yes" or "Likely no" (+ max 5 words reason)
Example: "Likely no; the trip was scary"
TYPE C - Choice questions (A or B):
Pattern: "Would X be more interested in A or B?"
→ Answer with just the chosen option + brief reason
Example: "The library; he loves reading"
TYPE D - Trait listing questions:
Pattern: "What personality traits...?"
→ List exact adjectives from context, comma-separated
Example: "Creative, patient, meticulous"
⚠️ Keep answer under 8 words total. NO long explanations.
Answer:"""
    elif category == 4:
        prompt = f"""Context:
{context}
Question: {question}
SPEAKER VERIFICATION:
- "Speaker Alex says:" → Alex's information
- "Speaker Ben says:" → Ben's information
- Match the question's subject to the correct speaker
ANSWER FORMAT:
- For emotion/reaction questions → Use complete short sentences
  Example: "He felt surprised and excited" (not "surprise and excitement")
- For factual questions → Noun phrases are OK
  Example: "The Eiffel Tower" | "Renewable energy sources"
EVIDENCE ASSESSMENT:
- If only a QUESTION exists without the answer in context → Still try to infer from surrounding context
- Only say "Not enough information" if truly impossible to determine
Complete answer:"""
    elif category == 5:
        if qa and hasattr(qa, 'adversarial_answer'):
            options = [qa.final_answer, qa.adversarial_answer]
            random.shuffle(options)
            options_str = f"'{options[0]}' or '{options[1]}'"
        else:
            options_str = "'Not mentioned in the conversation' or the correct answer based on the text"
        prompt = f"""Based on the context below, answer the question OR state it's not mentioned.
Context:
{context}
🚨 ADVERSARIAL CHECK (MANDATORY BEFORE ANSWERING):
Step 1: Who is the question asking about? Identify the SPECIFIC person (e.g., Alice or Bob)
Step 2: Scan ALL evidence - check the "Speaker [Name] says:" prefix for each memory
Step 3: Is there information about THIS SPECIFIC PERSON (not the other person)?
Step 4: If evidence only mentions the OTHER person → Answer "Not mentioned in the conversation"
Examples of TRAPS to avoid:
- Q asks about "Alice's project" but evidence shows "Speaker Bob says: my project..."
  → Answer: "Not mentioned in the conversation" (Alice's project ≠ Bob's project)
- Q asks about "What is Alice's favorite food" but evidence shows "Speaker Bob says: I think pizza is the best..."
  → Answer: "Not mentioned in the conversation" (This is Bob's opinion, not Alice's)
Question: {question}
Select the correct answer: {options_str}
Answer:"""
    else:
        prompt = f"""Based on a given context, please answer the question.
Context: {context}
Question: {question}
Short answer:"""
    return prompt


def build_category_prompt_fphm_style(category: int, context: str, question: str, qa=None) -> str:
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


def build_category_prompt_enhanced_oracle(category: int, context: str, question: str, qa=None) -> str:
    if category == 1:  # Single-hop
        prompt = f"""Context:
{context}
Question: {question}
Instructions:
1. Answer with 1-5 words only.
2. Use Arabic numerals for numbers.
3. Match the question's focus: answer the TYPE for "What kind", list ITEMS for "What items", and provide LOCATION for "Where".
4. Use generic terms where appropriate.
Short answer:"""
    elif category == 2:  # Temporal
        prompt = f"""Context:
{context}
Question: {question}
Instructions:
1. Analyze the question to determine if it asks for a specific date, a relative duration, or a duration length.
2. Use the 'talk start time' from the context for all date calculations.
3. Format your answer strictly according to the detected question type.
Answer:"""
    elif category == 3:  # Open-domain
        prompt = f"""Context:
{context}
Question: {question}
Instructions:
1. Detect the question type: single-word trait, yes/no likelihood, choice between A/B, or a list of traits.
2. Format your answer concisely based on the detected type.
3. Keep the entire answer under 8 words. Do not provide long explanations.
Answer:"""
    elif category == 4:  # Multi-hop
        prompt = f"""Context:
{context}
Question: {question}
Instructions:
1. Verify that the information you use comes from the correct speaker as specified in the question.
2. For questions about emotions or reactions, answer with a complete short sentence.
3. For factual questions, a noun phrase is sufficient.
Complete answer:"""
    elif category == 5:  # Adversarial
        if qa and hasattr(qa, 'adversarial_answer'):
            options = [qa.final_answer, qa.adversarial_answer]
            random.shuffle(options)
            options_str = f"'{options[0]}' or '{options[1]}'"
        else:
            options_str = "'Not mentioned in the conversation' or the correct answer"
        prompt = f"""Context:
{context}
Question: {question}
Instructions:
1. First, identify the specific person the question is about.
2. Scan all provided context to confirm the information is about that specific person.
3. If the context discusses a different person, you MUST answer 'Not mentioned in the conversation'.
Select the correct answer: {options_str}
Answer:"""
    else:  # Default
        prompt = f"""Based on the context, please answer the question.
Context: {context}
Question: {question}
Short answer:"""
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
            truncated_evidence = evidence_text
            analysis_str_builder.append(f"  Evidence Seen by Model (truncated):\n---\n{truncated_evidence}\n---\n")

        analysis_str_builder.append("\n--- Top 5 WORST Answers (by F1 Score) ---\n")
        for res in category_results[-5:][::-1]:
            analysis_str_builder.append(f"F1: {res['f1']:.4f}\n")
            analysis_str_builder.append(f"  Q: {res['question']}\n")
            analysis_str_builder.append(f"  A (Prediction): {res['prediction']}\n")
            analysis_str_builder.append(f"  A (Reference):  {res['reference']}\n")
            evidence_text = res.get('evidence_context', 'N/A')
            truncated_evidence = evidence_text
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


# --- MODIFICATION START ---
# 代码内注释：将原有的单样本处理逻辑封装成一个独立的“工作函数”。
# 这个函数接收所有必要的参数，并返回该样本的处理结果。这是并行化的关键一步。
def process_single_sample_oracle(args, sample, output_dir, base_run_name):
    llm_controller = LLMController(
        backend=args.backend, model=args.model,
        api_key=args.api_key, api_base=args.api_base
    )

    # 代码内注释：为每个样本创建独立的日志文件，避免并行写入时发生冲突。
    file_run_name = f"{base_run_name}_sample_{sample.sample_id}"
    logger = FPHMLogger(log_dir=output_dir, run_name=file_run_name)

    sample_metrics, sample_categories = [], []
    qa_results_for_analysis = []
    total_time_spent = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_qa_processed = 0

    all_turns_map = get_all_turns_map(sample)

    for qa in sample.qa:
        evidence_turn_ids = set(qa.evidence)
        if not evidence_turn_ids:
            continue

        context_turn_ids = set()
        if args.mode == 'oracle':
            context_turn_ids = evidence_turn_ids
        else:
            continue

        context_turn_objects = [all_turns_map[tid] for tid in context_turn_ids if tid in all_turns_map]
        context = build_context_from_turns(context_turn_objects)

        temperature = 0.1
        final_prompt = build_category_prompt_fphm_style(
            category=qa.category,
            context=context,
            question=qa.question,
            qa=qa
        )

        prompt_tokens = _estimate_tokens(final_prompt, args.model)
        total_prompt_tokens += prompt_tokens

        qa_start_time = time.time()
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

        qa_end_time = time.time()
        qa_duration = qa_end_time - qa_start_time
        total_time_spent += qa_duration

        completion_tokens = _estimate_tokens(prediction, args.model)
        total_completion_tokens += completion_tokens

        metrics = calculate_metrics(prediction, qa.final_answer)
        sample_metrics.append(metrics)
        sample_categories.append(qa.category)
        total_qa_processed += 1

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
            "mode": args.mode,
            "question": qa.question,
            "context_turn_ids": list(context_turn_ids),
            "final_prompt_to_llm": final_prompt,
            "prediction": prediction,
            "reference": qa.final_answer,
            "category": qa.category,
            "metrics": metrics,
            "duration_seconds": qa_duration,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        })

    # 代码内注释：函数返回所有需要聚合的数据。
    return (sample_metrics, sample_categories, qa_results_for_analysis,
            total_time_spent, total_prompt_tokens, total_completion_tokens, total_qa_processed)
# --- MODIFICATION END ---


def run_evidence_test(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_name = f"evidence_test_{args.mode}_{args.model.replace('/', '_')}"
    main_run_dir = os.path.join("fphm_runs", f"{base_run_name}_{timestamp}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"All outputs for this run will be saved in: {main_run_dir}")

    print("Loading LoCoMo dataset...")
    samples = load_locomo_dataset(args.dataset)

    all_metrics, all_categories = [], []
    all_qa_results_for_analysis = []
    total_time_spent = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_qa_processed = 0

# --- MODIFICATION START ---
    # 代码内注释：这是新的主控逻辑，负责创建进程池、分发任务并收集结果。
    if args.sample_index is not None:
        # 代码内注释：如果指定了 sample_index，则保持原有的单样本、单进程运行模式。
        if args.sample_index < len(samples):
            sample = samples[args.sample_index]
            print(f"Running in single-sample mode for sample index: {args.sample_index}")
            (metrics, categories, qa_results, time_spent, p_tokens, c_tokens, qa_count) = \
                process_single_sample_oracle(args, sample, main_run_dir, base_run_name)
            all_metrics.extend(metrics)
            all_categories.extend(categories)
            all_qa_results_for_analysis.extend(qa_results)
            total_time_spent += time_spent
            total_prompt_tokens += p_tokens
            total_completion_tokens += c_tokens
            total_qa_processed += qa_count
        else:
            print(f"Error: Sample index {args.sample_index} is out of bounds.")
            return
    else:
        # 代码内注释：如果没有指定 sample_index，则启动并行处理模式。
        print(f"Running in parallel mode for all {len(samples)} samples with {args.num_workers} workers.")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for sample in samples:
                # 代码内注释：为每个样本创建一个子目录，用于存放其独立的日志文件。
                sample_output_dir = os.path.join(main_run_dir, f"sample_{sample.sample_id}")
                os.makedirs(sample_output_dir, exist_ok=True)
                future = executor.submit(
                    process_single_sample_oracle, args, sample, sample_output_dir, base_run_name
                )
                futures.append(future)

            progress_bar = tqdm(as_completed(futures), total=len(samples), desc="Processing Samples")
            for future in progress_bar:
                try:
                    # 代码内注释：从每个完成的进程中收集返回的结果。
                    (s_metrics, s_categories, s_qa_results, s_time, s_p_tokens, s_c_tokens, s_qa_count) = future.result()
                    all_metrics.extend(s_metrics)
                    all_categories.extend(s_categories)
                    all_qa_results_for_analysis.extend(s_qa_results)
                    total_time_spent += s_time
                    total_prompt_tokens += s_p_tokens
                    total_completion_tokens += s_c_tokens
                    total_qa_processed += s_qa_count
                except Exception as e:
                    print(f"A sample process failed: {e}")
# --- MODIFICATION END ---

    if not all_metrics:
        print("No metrics were generated. Evaluation might have failed.")
        return

    aggregate_results = aggregate_metrics(all_metrics, all_categories)

    if total_qa_processed > 0:
        cost_analysis = {
            "total_qa_processed": total_qa_processed,
            "total_time_seconds": round(total_time_spent, 2),
            "avg_time_per_qa_seconds": round(total_time_spent / total_qa_processed, 2),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_prompt_tokens_per_qa": round(total_prompt_tokens / total_qa_processed),
            "avg_completion_tokens_per_qa": round(total_completion_tokens / total_qa_processed)
        }
        aggregate_results["cost_analysis"] = cost_analysis

    print(f"\n--- Evaluation Summary for Mode: {args.mode} ---")
    print(json.dumps(aggregate_results, indent=2))

    result_file = os.path.join(main_run_dir, f"results_{base_run_name}.json")
    with open(result_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    print(f"Results saved to {result_file}")

    detailed_results_file = os.path.join(main_run_dir, f"detailed_qa_results_{base_run_name}.json")
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_qa_results_for_analysis, f, indent=2, ensure_ascii=False)
    print(f"Detailed QA results saved to {detailed_results_file}")

    if all_qa_results_for_analysis:
        analyze_and_print_top_bottom_results(all_qa_results_for_analysis, main_run_dir, base_run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evidence-based testing on LoCoMo dataset.")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to the dataset file")
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--backend", type=str, help="LLM backend")
    parser.add_argument("--api_key", type=str, help="API key")
    parser.add_argument("--api_base", type=str, help="API base URL")
    parser.add_argument("--sample_index", type=int, default=None, help="Run on a single sample index. Omit to run on all.")
    parser.add_argument("--mode", type=str, choices=['oracle'], help="Testing mode.")
    parser.add_argument("--num-workers", type=int, default=5,
                        help="Number of parallel processes for multi-sample evaluation.")

    if len(sys.argv) == 1:
        print("No command-line arguments provided. Running with default quick-test settings...")
        args = argparse.Namespace(
            dataset="data/locomo10.json",
            model="gpt-4o-mini",
            backend="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            sample_index=None,#1
            mode='oracle',
            num_workers=5
        )
    else:
        print("Command-line arguments detected. Parsing as usual...")
        parser.set_defaults(
            model="gpt-4o-mini",
            backend="openai",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_API_BASE"),
            sample_index=None, # 代码内注释：修改默认值，当用户提供参数时，默认跑全量数据。
            mode='oracle'
        )
        args = parser.parse_args()

    run_evidence_test(args)
# --- fphm_code_changes_end ---
