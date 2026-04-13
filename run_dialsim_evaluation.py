import os

# Avoid HF network calls during long eval runs (models are expected to be cached).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import time
from datetime import datetime
from typing import List, Optional

from tqdm import tqdm

from dialsim_dataset import (
    DEFAULT_DIALSIM_SHOWS,
    DEFAULT_EASY_Q_TYPES,
    load_pickle_from_source,
    parse_script_to_turns,
    iter_easy_questions,
)
from memory_layer import LLMController
from fphm_core import FPHMSystem
import prompts


def generate_keyword_query(llm_controller: LLMController, original_query: str) -> dict:
    """Reuse HiGMem's query-rewriting prompt to generate a keyword query (optional)."""
    prompt = prompts.QUERY_REWRITING_PROMPT.format(original_query=original_query)
    schema = {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keyword_query": {"type": "string"},
                "profile_retrieval_keys": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["keyword_query", "profile_retrieval_keys"],
        },
    }
    try:
        response_str = llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": schema})
        data = json.loads(response_str)
        return {
            "keyword_query": data.get("keyword_query", " ".join(original_query.lower().replace("?", "").split())),
            "profile_retrieval_keys": data.get("profile_retrieval_keys", []),
        }
    except Exception:
        return {
            "keyword_query": " ".join(original_query.lower().replace("?", "").split()),
            "profile_retrieval_keys": [],
        }


def build_dialsim_mc_prompt(context: str, question: str, options: List[str]) -> str:
    options_block = "\n".join([f"- {opt}" for opt in options])
    # Keep this prompt identical across HiGMem/A-Mem DialSim eval scripts for fairness.
    return (
        "You are answering a multiple-choice question based on the given dialogue context.\n"
        "Select EXACTLY ONE option from the provided options.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_block}\n\n"
        "Rules:\n"
        "- Reply with the option text verbatim (copy it exactly).\n"
        "- Do NOT add any explanation.\n"
        '- If the answer cannot be inferred from the context, choose "I don\'t know. (None of the above)" '
        "IF it is present in the options.\n"
    ).strip()


def _normalize_for_contains(s: str) -> str:
    return (s or "").strip().lower()


def compute_answer_string_evidence_turn_ids(turn_ids: List[str], turn_texts: List[str], answer: str) -> List[str]:
    """Proxy 'evidence' for DialSim: turns whose text contains the gold answer string (case-insensitive)."""
    ans = _normalize_for_contains(answer)
    if not ans or ans == _normalize_for_contains("I don't know. (None of the above)"):
        return []
    hits = []
    for tid, txt in zip(turn_ids, turn_texts):
        if ans in _normalize_for_contains(txt):
            hits.append(tid)
    return hits


def run_scene(
    *,
    llm_controller: LLMController,
    scene_run_name: str,
    log_dir: str,
    turns: List[tuple],
    questions: List,
    args: argparse.Namespace,
):
    # Reset token counters at the start of each scene so we can attribute usage.
    try:
        llm_controller.llm.get_and_reset_token_counts()
    except Exception:
        pass

    memory_system = FPHMSystem(
        llm_controller=llm_controller,
        run_name=scene_run_name,
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
        ablation_mpnet_retrieval=args.ablation_mpnet_retrieval,
    )

    # Build memory
    mem_start = time.time()
    for idx, (speaker, text, timestamp) in enumerate(turns):
        turn_id = f"turn_{idx}"
        memory_system.add_turn(
            turn_id=turn_id,
            turn_content=text,
            speaker=speaker,
            timestamp=timestamp,
        )

    memory_system.finalize_memory_build()
    memory_system.build_indices()
    mem_end = time.time()

    mem_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    try:
        mem_tokens = llm_controller.llm.get_and_reset_token_counts()
    except Exception:
        pass

    # Answer questions
    records = []
    for q in questions:
        qa_start = time.time()

        if args.disable_query_rewriting_llm:
            keyword_query = q.question
            profile_keys = []
        else:
            qd = generate_keyword_query(llm_controller, q.question)
            keyword_query = qd["keyword_query"]
            profile_keys = qd["profile_retrieval_keys"]

        context, retrieval_trace = memory_system.retrieve_for_query(
            original_query=q.question,
            keyword_query=keyword_query,
            profile_retrieval_keys=profile_keys,
            k_profile=args.k_profile,
            k_event=args.k_event,
            k_turn=args.k_turn,
            return_trace=True,
        )

        prompt = build_dialsim_mc_prompt(context=context, question=q.question, options=q.options)
        answer_json = memory_system._get_llm_json_response(
            prompt,
            {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            caller="dialsim_final_answer_generation",
            temperature=args.temperature,
        )
        prediction = (answer_json or {}).get("answer", "") or ""

        qa_end = time.time()
        qa_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            qa_tokens = llm_controller.llm.get_and_reset_token_counts()
        except Exception:
            pass
        records.append(
            {
                "question": q.question,
                "options": q.options,
                "reference": q.answer,
                "prediction": prediction,
                "q_type": q.q_type,
                "q_id": q.q_id,
                "duration_seconds": qa_end - qa_start,
                "token_usage": qa_tokens,
                "retrieval_trace": retrieval_trace,
                "scene_memory_construction": {
                    "duration_seconds": mem_end - mem_start,
                    "token_usage": mem_tokens,
                },
            }
        )

    memory_system.executor.shutdown(wait=True)
    return records


def main():
    parser = argparse.ArgumentParser(description="Run HiGMem (FPHM) on DialSim (multiple-choice).")
    default_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HiGMem_Other", "dialsim_v1.1.zip"))
    parser.add_argument(
        "--dialsim_source",
        type=str,
        default=default_source,
        help="Path to DialSim zip (Deflate64) or extracted directory containing *.pickle files.",
    )
    parser.add_argument(
        "--shows",
        type=str,
        default=",".join(DEFAULT_DIALSIM_SHOWS),
        help="Comma-separated shows: friends,bigbang,theoffice",
    )
    parser.add_argument(
        "--question_types",
        type=str,
        default=",".join(DEFAULT_EASY_Q_TYPES),
        help="Comma-separated easy question buckets to evaluate.",
    )
    parser.add_argument("--max_scenes", type=int, default=2, help="Debug limit. -1 for all scenes.")
    parser.add_argument("--max_questions_per_scene", type=int, default=5, help="Debug limit. -1 for all.")

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--backend", type=str, default="openai")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--api_base",
        type=str,
        default=os.getenv("OPENAI_API_BASE"),
        help="API base URL (end with /v1).",
    )

    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--disable_query_rewriting_llm", action="store_true", help="Do not call LLM to rewrite queries.")

    # HiGMem retrieval params (same flags as run_fphm_evaluation.py)
    parser.add_argument("--ablation-no-profile", action="store_true", help="Disable character profiles.")
    parser.add_argument("--ablation-event-title-only", action="store_true")
    parser.add_argument("--ablation-event-metadata-only", action="store_true")
    parser.add_argument("--ablation-attribute-profile", action="store_true")
    parser.add_argument("--ablation-no-fact-judgment", action="store_true")
    parser.add_argument("--ablation-no-filter", action="store_true")
    parser.add_argument("--ablation-no-link", action="store_true")
    parser.add_argument("--ablation-no-event", action="store_true")
    parser.add_argument("--ablation-mpnet-retrieval", action="store_true")
    parser.add_argument("--k_profile", type=int, default=3)
    # Paper default is k_event=10 (previously hard-wired to 7 by mistake).
    parser.add_argument("--k_event", type=int, default=10)
    parser.add_argument("--k_turn", type=int, default=10)

    args = parser.parse_args()

    shows = [s.strip() for s in args.shows.split(",") if s.strip()]
    q_types = [s.strip() for s in args.question_types.split(",") if s.strip()]

    # third_party path for Deflate64 zip support
    third_party_dir = os.path.join(os.path.dirname(__file__), "third_party", "zipfile_deflate64")

    llm_controller = LLMController(
        backend=args.backend,
        model=args.model,
        api_key=args.api_key,
        api_base=args.api_base,
    )

    run_name = f"{args.model.replace('/', '_')}_{args.backend}_dialsim"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "dialsim_runs", f"{run_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")
    meta_path = os.path.join(out_dir, "run_meta.json")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dialsim_source": os.path.abspath(args.dialsim_source),
                "shows": shows,
                "question_types": q_types,
                "max_scenes": args.max_scenes,
                "max_questions_per_scene": args.max_questions_per_scene,
                "model": args.model,
                "backend": args.backend,
                "k_event": args.k_event,
                "k_turn": args.k_turn,
                "ablation": {
                    "no_profile": args.ablation_no_profile,
                    "event_meta_only": args.ablation_event_metadata_only,
                    "no_event": args.ablation_no_event,
                    "no_filter": args.ablation_no_filter,
                    "no_link": args.ablation_no_link,
                },
                "prompt_note": "DialSim is multiple-choice; final QA prompt is shared with AgenticMemory for fairness.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    total_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for show in shows:
            show_pickle = f"{show}_dialsim.pickle"
            show_data = load_pickle_from_source(args.dialsim_source, show_pickle, third_party_dir=third_party_dir)

            scenes_processed = 0
            for episode_name, episode in tqdm(show_data.items(), desc=f"{show}:episodes"):
                # episode is a dict: scene_id(int) -> {date, script, easy_q, hard_q}
                for scene_id, scene_item in episode.items():
                    if args.max_scenes != -1 and scenes_processed >= args.max_scenes:
                        break
                    scenes_processed += 1

                    script = str(scene_item.get("script", "") or "")
                    date = str(scene_item.get("date", "") or "")
                    parsed_turns = parse_script_to_turns(script)
                    turns = [(spk, txt, f"{date}#{i:04d}") for i, (spk, txt) in enumerate(parsed_turns)]

                    # Build question list
                    qs = list(
                        iter_easy_questions(
                            show=show,
                            episode=episode_name,
                            scene_id=int(scene_id),
                            scene_item=scene_item,
                            include_q_types=q_types,
                        )
                    )
                    if args.max_questions_per_scene != -1:
                        qs = qs[: args.max_questions_per_scene]

                    if not qs:
                        continue

                    scene_run_name = f"{run_name}_{show}_{episode_name.replace(' ', '_')}_scene{scene_id}"
                    log_dir = out_dir  # keep logs under the run dir
                    scene_records = run_scene(
                        llm_controller=llm_controller,
                        scene_run_name=scene_run_name,
                        log_dir=log_dir,
                        turns=turns,
                        questions=qs,
                        args=args,
                    )

                    # Add stable metadata per record (avoid storing full scripts)
                    turn_ids = [f"turn_{i}" for i in range(len(turns))]
                    turn_texts = [t[1] for t in turns]
                    for q_obj, rec in zip(qs, scene_records):
                        rec.update(
                            {
                                "show": show,
                                "episode": episode_name,
                                "scene_id": int(scene_id),
                                "date": date,
                                "evidence_turn_ids_by_answer_string": compute_answer_string_evidence_turn_ids(
                                    turn_ids, turn_texts, q_obj.answer
                                ),
                                "num_turns_in_scene": len(turns),
                            }
                        )
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        total_written += 1

                if args.max_scenes != -1 and scenes_processed >= args.max_scenes:
                    break

    print(f"Wrote {total_written} QA records to: {out_path}")
    print(f"Meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
