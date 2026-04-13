import os

# Avoid HF network calls during long eval runs (models are expected to be cached locally).
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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np

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


def _cosine_sim_scores(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between query_vec (d,) and mat (n,d)."""
    if mat.size == 0:
        return np.array([], dtype=np.float32)
    q = query_vec.astype(np.float32, copy=False)
    m = mat.astype(np.float32, copy=False)
    q_norm = float(np.linalg.norm(q) + 1e-8)
    m_norm = np.linalg.norm(m, axis=1) + 1e-8
    return (m @ q) / (m_norm * q_norm)


def fast_retrieve_for_query(
    memory_system: FPHMSystem,
    original_query: str,
    keyword_query: str,
    *,
    k_event: int,
    k_turn: int,
    k_turn_per_event: int,
    max_candidate_turns: int,
    return_trace: bool = False,
) -> Tuple[str, dict]:
    """
    Faster DialSim-oriented retrieval:
    - Retrieve top-k events by embedding.
    - Within each event, select top-m turns by embedding similarity (no per-event LLM call).
    - Optionally expand by turn links (same behavior as core).
    - Run a single LLM filtering pass over capped candidate turns (unless no-filter ablation).

    This preserves the Event->Turn two-layer idea but avoids k_event LLM calls per question,
    which is critical for finishing DialSim within tight time budgets.
    """
    # --- Stage 1: vector recall ---
    turn_indices = memory_system.turn_retriever.search(keyword_query, k=k_turn)
    candidate_turn_ids = [memory_system.turn_retriever.document_ids[i] for i in turn_indices]

    event_indices = memory_system.event_retriever.search(keyword_query, k=k_event)
    candidate_event_ids = [memory_system.event_retriever.document_ids[i] for i in event_indices]

    # --- Stage 2: within-event embedding selection ---
    event_to_top_turns = {}
    selected_turn_ids = set(candidate_turn_ids)

    try:
        query_vec = memory_system.turn_retriever.model.encode([keyword_query])[0]
    except Exception:
        query_vec = None

    id_to_index = getattr(memory_system.turn_retriever, "id_to_index", {}) or {}
    for eid in candidate_event_ids:
        event = getattr(memory_system, "events", {}).get(eid)
        if not event:
            continue
        turn_ids = [tid for tid in (event.turn_note_ids or []) if tid in id_to_index]
        if not turn_ids:
            continue
        if query_vec is None or memory_system.turn_retriever.embeddings is None:
            chosen = turn_ids[-max(0, int(k_turn_per_event)) :]
        else:
            idxs = [id_to_index[tid] for tid in turn_ids]
            emb = memory_system.turn_retriever.embeddings[idxs]
            scores = _cosine_sim_scores(np.asarray(query_vec), np.asarray(emb))
            m = min(int(k_turn_per_event), len(turn_ids))
            top_local = np.argsort(scores)[-m:][::-1].tolist()
            chosen = [turn_ids[i] for i in top_local]
        event_to_top_turns[eid] = chosen
        selected_turn_ids.update(chosen)

    # --- Stage 3: optional link expansion (feature in core code) ---
    if not getattr(memory_system, "ablation_no_link", False):
        for tid in list(selected_turn_ids):
            note = getattr(memory_system, "turn_notes", {}).get(tid)
            if not note:
                continue
            try:
                linked = [lnk.target_id for lnk in (note.links or [])]
            except Exception:
                linked = []
            for lid in linked:
                if lid:
                    selected_turn_ids.add(lid)

    fused_ids = [tid for tid in selected_turn_ids if tid in getattr(memory_system, "turn_notes", {})]

    # --- Stage 4: cap candidates by similarity ---
    if (
        max_candidate_turns is not None
        and len(fused_ids) > int(max_candidate_turns)
        and query_vec is not None
        and memory_system.turn_retriever.embeddings is not None
    ):
        try:
            idxs = [id_to_index[tid] for tid in fused_ids if tid in id_to_index]
            emb = memory_system.turn_retriever.embeddings[idxs]
            scores = _cosine_sim_scores(np.asarray(query_vec), np.asarray(emb))
            n = min(int(max_candidate_turns), len(idxs))
            top = np.argsort(scores)[-n:][::-1].tolist()
            fused_ids = [fused_ids[i] for i in top]
        except Exception:
            fused_ids = fused_ids[: int(max_candidate_turns)]

    # --- Stage 5: LLM filtering (single pass) ---
    if getattr(memory_system, "ablation_no_filter", False):
        relevant_turn_ids = list(fused_ids)
    else:
        candidate_turns = {tid: memory_system.turn_notes[tid].content for tid in fused_ids if tid in memory_system.turn_notes}
        try:
            relevant_turn_ids = memory_system._judge_relevance_sequential(original_query, candidate_turns, "turn")
        except Exception:
            relevant_turn_ids = []
        # Guard against hallucinated IDs.
        relevant_turn_ids = [tid for tid in relevant_turn_ids if tid in candidate_turns]

    # --- Stage 6: build context (match core formatting) ---
    try:
        sorted_turns = sorted(
            [memory_system.turn_notes[tid] for tid in relevant_turn_ids if tid in memory_system.turn_notes],
            key=lambda t: t.timestamp,
        )
    except Exception:
        sorted_turns = [memory_system.turn_notes[tid] for tid in relevant_turn_ids if tid in memory_system.turn_notes]
        sorted_turns.sort(key=lambda t: t.id)

    final_context_parts = []
    for t in sorted_turns:
        final_context_parts.append(
            (
                f"--- Turn Start ---\n"
                f"Timestamp: {t.timestamp}\n"
                f"Speaker: {t.speaker}\n"
                f"Content: {t.content}\n"
                f"Context Summary: {t.context}\n"
                f"--- Turn End ---"
            )
        )
    final_context = "\n\n".join(final_context_parts)

    trace = {}
    if return_trace:
        trace = {
            "mode": "fast_event_turn",
            "keyword_query": keyword_query,
            "candidate_turn_ids": candidate_turn_ids,
            "candidate_event_ids": candidate_event_ids,
            "event_to_top_turns": event_to_top_turns,
            "final_fused_candidate_turn_ids": fused_ids,
            "relevant_turn_ids": relevant_turn_ids,
            "k_turn_per_event": int(k_turn_per_event),
            "max_candidate_turns": int(max_candidate_turns),
            "llm_filter_enabled": (not getattr(memory_system, "ablation_no_filter", False)),
        }
    return final_context, trace


def _load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HiGMem (FPHM) on DialSim with a streaming (turn-level) manifest.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to a DialSim streaming manifest JSON.")
    parser.add_argument(
        "--shows",
        type=str,
        default=None,
        help="Optional comma-separated subset of shows to run (e.g., friends,bigbang). "
             "Default: run all shows present in the manifest.",
    )

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

    parser.add_argument(
        "--parallel_shows",
        action="store_true",
        help="Run each show timeline in parallel (threaded). This matches locomo10's 'parallel across samples' behavior "
        "and reduces wall time without changing per-show semantics.",
    )
    parser.add_argument(
        "--show_workers",
        type=int,
        default=3,
        help="Max number of show threads when --parallel_shows is enabled.",
    )

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
    parser.add_argument(
        "--use-fast-retrieval",
        action="store_true",
        help="(NOT for paper reproduction) Use a faster Event->Turn retrieval that avoids per-event LLM selection. "
        "Default uses the original locomo10-consistent retrieval (FPHMSystem.retrieve_for_query).",
    )
    # Backward-compat: this flag used to be required to enable the original path. Now original is the default.
    parser.add_argument("--use-original-retrieval", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--k_turn_per_event", type=int, default=5, help="Fast retrieval: top turns per recalled event.")
    parser.add_argument("--max_candidate_turns", type=int, default=50, help="Fast retrieval: cap candidate turns before LLM filtering.")
    parser.add_argument("--progress_every_scenes", type=int, default=10, help="Print progress every N scenes.")

    args = parser.parse_args()

    manifest = _load_manifest(args.manifest)
    shows = list((manifest.get("shows") or {}).keys())
    if not shows:
        raise ValueError("Manifest has no shows.")
    if args.shows:
        wanted = {s.strip() for s in str(args.shows).split(",") if s.strip()}
        shows = [s for s in shows if s in wanted]
        if not shows:
            raise ValueError(f"--shows filtered everything out. wanted={sorted(wanted)}; manifest_shows={sorted((manifest.get('shows') or {}).keys())}")

    qrw_tag = "noqrw" if args.disable_query_rewriting_llm else "qrw"
    run_name = f"{args.model.replace('/', '_')}_{args.backend}_dialsim_stream_kevent{int(args.k_event)}_{qrw_tag}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "dialsim_runs", f"{run_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "predictions.jsonl")
    meta_path = os.path.join(out_dir, "run_meta.json")

    if args.use_original_retrieval:
        print("NOTE: --use-original-retrieval is now the default; you can omit it.")

    use_fast_retrieval = bool(args.use_fast_retrieval) and (not bool(args.use_original_retrieval))

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": os.path.abspath(args.manifest),
                "dialsim_version": manifest.get("dialsim_version"),
                "seed": manifest.get("seed"),
                "targets": manifest.get("targets"),
                "shows": shows,
                "model": args.model,
                "backend": args.backend,
                "disable_query_rewriting_llm": bool(args.disable_query_rewriting_llm),
                "parallel_shows": bool(args.parallel_shows),
                "show_workers": int(args.show_workers),
                "k_profile": args.k_profile,
                "k_event": args.k_event,
                "k_turn": args.k_turn,
                "retrieval_mode": "fast_event_turn" if use_fast_retrieval else "original",
                "k_turn_per_event": int(args.k_turn_per_event),
                "max_candidate_turns": int(args.max_candidate_turns),
                "ablation": {
                    "no_profile": args.ablation_no_profile,
                    "event_title_only": args.ablation_event_title_only,
                    "event_metadata_only": args.ablation_event_metadata_only,
                    "attribute_profile": args.ablation_attribute_profile,
                    "no_fact_judgment": args.ablation_no_fact_judgment,
                    "no_filter": args.ablation_no_filter,
                    "no_link": args.ablation_no_link,
                    "no_event": args.ablation_no_event,
                    "mpnet_retrieval": args.ablation_mpnet_retrieval,
                },
                "prompt_note": "DialSim is multiple-choice; final QA prompt is shared with AgenticMemory for fairness.",
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    def _run_single_show(show: str) -> Tuple[str, int, str]:
        """Run a single show and write to a per-show JSONL file; returns (show, written, path)."""
        show_block = (manifest.get("shows") or {}).get(show) or {}
        scenes = show_block.get("scenes") or []
        if not isinstance(scenes, list) or not scenes:
            return show, 0, ""

        llm_controller = LLMController(
            backend=args.backend,
            model=args.model,
            api_key=args.api_key,
            api_base=args.api_base,
        )

        show_run_name = f"{run_name}_{show}"
        memory_system = FPHMSystem(
            llm_controller=llm_controller,
            run_name=show_run_name,
            log_dir=out_dir,
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

        out_show_path = os.path.join(out_dir, f"predictions_{show}.jsonl")
        written = 0
        run_start = time.time()

        turn_ids_seen: List[str] = []
        turn_texts_seen: List[str] = []

        with open(out_show_path, "w", encoding="utf-8") as out_f:
            for scene_idx, scene in enumerate(scenes):
                turns = scene.get("turns") or []
                questions = scene.get("questions") or []

                # --- memory construction for this scene's turns ---
                try:
                    llm_controller.llm.get_and_reset_token_counts()
                except Exception:
                    pass
                mem_start = time.time()

                for t in turns:
                    turn_id = str(t.get("turn_id", "") or "")
                    speaker = str(t.get("speaker", "") or "")
                    text = str(t.get("text", "") or "")
                    ts = str(t.get("timestamp", "") or "")
                    if not turn_id:
                        turn_id = f"{show}_scene{scene_idx}_turn{len(turn_ids_seen)}"
                    memory_system.add_turn(turn_id=turn_id, turn_content=text, speaker=speaker, timestamp=ts)
                    turn_ids_seen.append(turn_id)
                    turn_texts_seen.append(text)

                mem_end = time.time()
                mem_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                try:
                    mem_tokens = llm_controller.llm.get_and_reset_token_counts()
                except Exception:
                    pass

                # --- scene-end QA ---
                if not isinstance(questions, list) or not questions:
                    continue

                for q in questions:
                    question_text = str(q.get("question", "") or "")
                    options = q.get("options", []) or []
                    if not isinstance(options, list):
                        options = []
                    options = [str(x) for x in options]
                    reference = str(q.get("answer", "") or "")

                    try:
                        llm_controller.llm.get_and_reset_token_counts()
                    except Exception:
                        pass
                    qa_start = time.time()

                    if args.disable_query_rewriting_llm:
                        keyword_query = question_text
                        profile_keys = []
                    else:
                        qd = generate_keyword_query(llm_controller, question_text)
                        keyword_query = qd["keyword_query"]
                        profile_keys = qd["profile_retrieval_keys"]

                    if not use_fast_retrieval:
                        context, retrieval_trace = memory_system.retrieve_for_query(
                            original_query=question_text,
                            keyword_query=keyword_query,
                            profile_retrieval_keys=profile_keys,
                            k_profile=args.k_profile,
                            k_event=args.k_event,
                            k_turn=args.k_turn,
                            return_trace=True,
                        )
                    else:
                        context, retrieval_trace = fast_retrieve_for_query(
                            memory_system,
                            original_query=question_text,
                            keyword_query=keyword_query,
                            k_event=args.k_event,
                            k_turn=args.k_turn,
                            k_turn_per_event=args.k_turn_per_event,
                            max_candidate_turns=args.max_candidate_turns,
                            return_trace=True,
                        )

                    prompt = build_dialsim_mc_prompt(context=context, question=question_text, options=options)
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

                    record = {
                        "show": show,
                        "episode": scene.get("episode"),
                        "scene_id": scene.get("scene_id"),
                        "date": scene.get("date"),
                        "scene_uid": scene.get("scene_uid"),
                        "scene_is_partial": bool(scene.get("is_partial", False)),
                        "scene_index_in_show": int(scene_idx),
                        "memory_state": {
                            "turns_seen": len(turn_ids_seen),
                            "events_seen": len(getattr(memory_system, "events", {}) or {}),
                            "profiles_seen": len(getattr(memory_system, "profiles", {}) or {}),
                        },
                        "question": question_text,
                        "options": options,
                        "reference": reference,
                        "prediction": prediction,
                        "q_type": q.get("q_type"),
                        "q_id": q.get("q_id"),
                        "split": q.get("split"),
                        "duration_seconds": qa_end - qa_start,
                        "token_usage": qa_tokens,
                        "retrieval_trace": retrieval_trace,
                        "evidence_turn_ids_by_answer_string": compute_answer_string_evidence_turn_ids(
                            turn_ids_seen, turn_texts_seen, reference
                        ),
                        "num_turns_in_scene": len(turns),
                        "scene_memory_construction": {
                            "duration_seconds": mem_end - mem_start,
                            "token_usage": mem_tokens,
                            "num_turns_added": len(turns),
                        },
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

                if args.progress_every_scenes and ((scene_idx + 1) % int(args.progress_every_scenes) == 0):
                    elapsed = time.time() - run_start
                    print(
                        f"[{show}] scene {scene_idx + 1}/{len(scenes)} "
                        f"turns_seen={len(turn_ids_seen)} q_written={written} "
                        f"elapsed_h={elapsed / 3600:.2f}",
                        flush=True,
                    )

        memory_system.executor.shutdown(wait=True)
        return show, written, out_show_path

    show_results: Dict[str, Tuple[int, str]] = {}
    if args.parallel_shows and len(shows) > 1:
        max_workers = max(1, min(int(args.show_workers), len(shows)))
        print(f"Running shows in parallel (threads): shows={shows} show_workers={max_workers}", flush=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_run_single_show, s): s for s in shows}
            for fut in as_completed(futs):
                show, written, path = fut.result()
                show_results[show] = (written, path)
    else:
        for s in shows:
            show, written, path = _run_single_show(s)
            show_results[show] = (written, path)

    # Merge per-show outputs into the canonical predictions.jsonl (stable order by `shows` list).
    total_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for s in shows:
            written, path = show_results.get(s, (0, ""))
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
            total_written += int(written)

    print(f"Wrote {total_written} QA records to: {out_path}")
    print(f"Meta saved to: {meta_path}")


if __name__ == "__main__":
    main()
