import argparse
import json
import pickle
import re
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


def _utf8_len(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, (bytes, bytearray)):
        return len(x)
    if isinstance(x, str):
        return len(x.encode("utf-8"))
    if isinstance(x, (int, float, bool)):
        return len(str(x).encode("utf-8"))
    # Fallback: JSON-ish string
    try:
        return len(json.dumps(x, ensure_ascii=False).encode("utf-8"))
    except Exception:
        return len(str(x).encode("utf-8"))


def _bytes_for_str_list(items: Any) -> int:
    if not items:
        return 0
    total = 0
    for it in items:
        total += _utf8_len(it)
    return total


def _bytes_for_turn_note(tn: Any) -> int:
    # TurnNote dataclass fields (H-Mem/fphm_structures.py)
    total = 0
    total += _utf8_len(getattr(tn, "id", ""))
    total += _utf8_len(getattr(tn, "speaker", ""))
    total += _utf8_len(getattr(tn, "timestamp", ""))
    total += _utf8_len(getattr(tn, "content", ""))
    total += _utf8_len(getattr(tn, "context", ""))
    total += _bytes_for_str_list(getattr(tn, "keywords", None))
    total += _bytes_for_str_list(getattr(tn, "tags", None))
    # Links and parent ids
    total += _utf8_len([getattr(l, "target_id", "") for l in getattr(tn, "links", [])])
    total += _utf8_len([getattr(l, "relationship_type", "") for l in getattr(tn, "links", [])])
    total += _utf8_len(getattr(tn, "parent_event_ids", None))
    return total


def _bytes_for_fact_sheet(fs: Any) -> int:
    if fs is None:
        return 0
    total = 0
    timeline = getattr(fs, "timeline", None) or []
    for item in timeline:
        # item is a dict like {"timestamp": "...", "fact": "...", "evidence_turn_id": "..."}
        total += _utf8_len(item)
    total += _utf8_len(getattr(fs, "key_entities", None))
    return total


def _bytes_for_event(ev: Any) -> int:
    # EventSummary dataclass fields (H-Mem/fphm_structures.py)
    total = 0
    total += _utf8_len(getattr(ev, "id", ""))
    total += _utf8_len(getattr(ev, "title", ""))
    total += _utf8_len(getattr(ev, "summary_content", ""))
    total += _bytes_for_fact_sheet(getattr(ev, "fact_sheet", None))
    total += _bytes_for_str_list(getattr(ev, "keywords", None))
    total += _bytes_for_str_list(getattr(ev, "tags", None))
    total += _utf8_len(getattr(ev, "turn_note_ids", None))
    total += _utf8_len([getattr(l, "target_id", "") for l in getattr(ev, "links", [])])
    total += _utf8_len([getattr(l, "relationship_type", "") for l in getattr(ev, "links", [])])
    total += _utf8_len(getattr(ev, "specific_entities", None))
    return total


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + 1e-12)


def _bench_topk_search_ms(emb_norm: np.ndarray, k: int, reps: int = 50, seed: int = 0) -> float:
    n, dim = emb_norm.shape
    if n == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(dim, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    k = min(k, n)

    # Warmup
    sims = emb_norm @ q
    _ = np.argpartition(sims, -k)[-k:]

    start = time.perf_counter()
    for _ in range(reps):
        sims = emb_norm @ q
        _ = np.argpartition(sims, -k)[-k:]
    end = time.perf_counter()
    return (end - start) * 1000.0 / reps


def _load_locomo_turn_ids_in_order(dataset_path: Path) -> Dict[int, List[str]]:
    """Turn order = the same order used by both systems when constructing memories."""
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    out: Dict[int, List[str]] = {}
    for sample_idx, sample in enumerate(data):
        conv = sample.get("conversation", {})
        turn_ids: List[str] = []
        for key, value in conv.items():
            if not (isinstance(key, str) and key.startswith("session_")):
                continue
            if not isinstance(value, list):
                continue
            for turn in value:
                tid = turn.get("dia_id")
                if tid is not None:
                    turn_ids.append(str(tid))
        out[sample_idx] = turn_ids
    return out


def _load_higmem_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    with open(checkpoint_path, "rb") as f:
        return pickle.load(f)


def _extract_retriever_arrays(retriever_obj: Any) -> Tuple[List[str], np.ndarray]:
    doc_ids = getattr(retriever_obj, "document_ids", None)
    emb = getattr(retriever_obj, "embeddings", None)
    if doc_ids is None or emb is None:
        return [], np.zeros((0, 0), dtype=np.float32)
    return list(doc_ids), np.asarray(emb)


def _load_amem_embeddings(amem_cache_dir: Path, sample_id: int) -> np.ndarray:
    emb_path = amem_cache_dir / f"retriever_cache_embeddings_sample_{sample_id}.npy"
    return np.load(str(emb_path))


def _load_amem_memories(amem_cache_dir: Path, sample_id: int) -> List[Any]:
    mem_path = amem_cache_dir / f"memory_and_stats_cache_sample_{sample_id}.pkl"
    with open(mem_path, "rb") as f:
        cached = pickle.load(f)
    mem_dict = cached.get("memories", {})
    # dict preserves insertion order; A-Mem retrieval uses list(self.memories.values()).
    return list(mem_dict.values())

def _resolve_higmem_run_dir(arg_path: Path, config_name: str) -> Path:
    """
    exp5 needs a specific HiGMem run directory (contains sample_*/checkpoints/...).
    To reduce manual mistakes, we also accept the runs root (e.g., `fphm_runs`) and
    automatically pick the newest matching run dir for the given config_name.
    """
    if (arg_path / "sample_0").exists():
        return arg_path
    if not arg_path.exists():
        raise FileNotFoundError(str(arg_path))
    pat = re.compile(re.escape(config_name) + r"_(\d{8}_\d{6})$")
    candidates: List[Tuple[str, Path]] = []
    for d in arg_path.iterdir():
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue
        # Keep only dirs that look like full runs.
        if not list(d.glob("sample_*")):
            continue
        candidates.append((m.group(1), d))
    if not candidates:
        raise FileNotFoundError(f"No HiGMem run dir found under {arg_path} for config_name={config_name}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp5: Scaling analysis (storage + vector retrieval latency).")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json", help="Path to locomo10.json")
    parser.add_argument(
        "--higmem_run_dir",
        type=str,
        default="fphm_runs",
        help="HiGMem run dir (contains sample_*/checkpoints/*.pkl) OR the runs root directory (e.g., fphm_runs). "
             "If a root is provided, the newest matching run is selected automatically.",
    )
    parser.add_argument(
        "--higmem_config_name",
        type=str,
        default="gpt-4o-mini_event_meta_no_profile_kevent10_qrw_sync",
        help="Config name used in checkpoint filenames.",
    )
    parser.add_argument(
        "--higmem_k_event",
        type=int,
        default=None,
        help="Top-K for HiGMem event retrieval used for vector-only latency estimates. "
             "If omitted, inferred from `--higmem_config_name` (pattern: kevent{K}); defaults to 10 if not found.",
    )
    parser.add_argument(
        "--amem_cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "AgenticMemory" / "cached_memories_advanced_openai_gpt-4o-mini"),
        help="A-Mem cache dir containing retriever_cache_embeddings_sample_*.npy and memory_and_stats_cache_sample_*.pkl",
    )
    parser.add_argument("--prefix_turns", type=str, default="10,50,100,200", help="Comma-separated prefix sizes (turns).")
    parser.add_argument("--output_csv", type=str, default="analysis_results/exp5_scaling_locomo10.csv")
    parser.add_argument(
        "--simulate_sizes",
        type=str,
        default="",
        help="Optional: comma-separated synthetic sizes (e.g., 1000,10000,100000). "
             "Uses random embeddings to estimate vector-search scaling; no LLM calls.",
    )
    parser.add_argument("--simulate_embed_dim", type=int, default=384, help="Embedding dim for synthetic scaling.")
    parser.add_argument(
        "--simulate_event_ratio",
        type=float,
        default=0.085,
        help="For synthetic HiGMem, approximate event_count ~= turns * ratio.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(str(dataset_path))

    higmem_run_dir = _resolve_higmem_run_dir(Path(args.higmem_run_dir), args.higmem_config_name)
    amem_cache_dir = Path(args.amem_cache_dir)

    if args.higmem_k_event is None:
        m = re.search(r"kevent(\d+)", str(args.higmem_config_name))
        args.higmem_k_event = int(m.group(1)) if m else 10

    prefix_sizes = [int(x.strip()) for x in args.prefix_turns.split(",") if x.strip()]
    prefix_sizes = sorted(set(prefix_sizes))

    turn_order_by_sample = _load_locomo_turn_ids_in_order(dataset_path)

    rows: List[Dict[str, Any]] = []

    for sample_id in sorted(turn_order_by_sample.keys()):
        turn_ids_order = turn_order_by_sample[sample_id]
        num_turns_total = len(turn_ids_order)

        # --- HiGMem (full system checkpoint) ---
        chk_path = higmem_run_dir / f"sample_{sample_id}" / "checkpoints" / f"checkpoint_{args.higmem_config_name}_sample_{sample_id}_final.pkl"
        if not chk_path.exists():
            raise FileNotFoundError(str(chk_path))
        state = _load_higmem_checkpoint(chk_path)

        turn_notes: Dict[str, Any] = state.get("turn_notes", {}) or {}
        events: Dict[str, Any] = state.get("events", {}) or {}

        turn_doc_ids, turn_emb = _extract_retriever_arrays(state.get("turn_retriever"))
        event_doc_ids, event_emb = _extract_retriever_arrays(state.get("event_retriever"))

        turn_id_to_row = {tid: i for i, tid in enumerate(turn_doc_ids)}
        event_id_to_row = {eid: i for i, eid in enumerate(event_doc_ids)}

        # Build turn embeddings aligned with dataset order (prefix slicing becomes trivial).
        ordered_turn_rows = []
        for tid in turn_ids_order:
            if tid in turn_id_to_row:
                ordered_turn_rows.append(turn_id_to_row[tid])
        ordered_turn_rows = np.asarray(ordered_turn_rows, dtype=np.int64)
        ordered_turn_emb = turn_emb[ordered_turn_rows] if ordered_turn_rows.size else np.zeros((0, 0), dtype=np.float32)
        ordered_turn_emb_norm = _normalize_rows(ordered_turn_emb) if ordered_turn_emb.size else ordered_turn_emb

        # Precompute event "first turn index" in conversation order.
        turn_pos = {tid: idx for idx, tid in enumerate(turn_ids_order)}
        event_first_pos: Dict[str, int] = {}
        for eid, ev in events.items():
            poss = [turn_pos.get(tid) for tid in getattr(ev, "turn_note_ids", []) if tid in turn_pos]
            if not poss:
                continue
            event_first_pos[eid] = min(p for p in poss if p is not None)

        # Pre-normalize all event embeddings (for large sizes this saves time).
        event_emb_norm_all = _normalize_rows(event_emb) if event_emb.size else event_emb

        # --- A-Mem caches ---
        amem_emb_all = _load_amem_embeddings(amem_cache_dir, sample_id)
        amem_emb_norm_all = _normalize_rows(amem_emb_all) if amem_emb_all.size else amem_emb_all
        amem_memories_all = _load_amem_memories(amem_cache_dir, sample_id)

        for n in prefix_sizes:
            n_turns = min(n, num_turns_total)

            # ---- HiGMem prefix ----
            # Turn notes
            turn_ids_prefix = turn_ids_order[:n_turns]
            turn_text_bytes = 0
            for tid in turn_ids_prefix:
                tn = turn_notes.get(tid)
                if tn is None:
                    continue
                turn_text_bytes += _bytes_for_turn_note(tn)
            turn_emb_bytes = 0
            if ordered_turn_emb.shape[0] >= n_turns and ordered_turn_emb.ndim == 2:
                turn_emb_bytes = int(n_turns * ordered_turn_emb.shape[1] * 4)

            # Events that "exist" by this prefix (approx: earliest affiliated turn is within prefix).
            event_ids_prefix = [eid for eid, pos in event_first_pos.items() if pos < n_turns]
            event_text_bytes = 0
            for eid in event_ids_prefix:
                ev = events.get(eid)
                if ev is None:
                    continue
                event_text_bytes += _bytes_for_event(ev)
            event_emb_bytes = 0
            if event_emb.shape[1] if event_emb.ndim == 2 else 0:
                dim = int(event_emb.shape[1]) if event_emb.ndim == 2 else 0
                event_emb_bytes = int(len(event_ids_prefix) * dim * 4)

            higmem_storage_mb = (turn_text_bytes + turn_emb_bytes + event_text_bytes + event_emb_bytes) / 1_000_000.0

            # Vector retrieval latency: event search + turn search (excluding LLM calls).
            higmem_turn_ms = _bench_topk_search_ms(ordered_turn_emb_norm[:n_turns], k=10) if n_turns else 0.0
            # Build event prefix embedding matrix
            event_rows = [event_id_to_row[eid] for eid in event_ids_prefix if eid in event_id_to_row]
            if event_rows:
                event_rows_arr = np.asarray(event_rows, dtype=np.int64)
                event_prefix_norm = event_emb_norm_all[event_rows_arr]
                higmem_event_ms = _bench_topk_search_ms(event_prefix_norm, k=int(args.higmem_k_event))
            else:
                higmem_event_ms = 0.0
            higmem_retrieval_ms = higmem_turn_ms + higmem_event_ms

            # ---- A-Mem prefix ----
            amem_turn_emb_bytes = int(n_turns * amem_emb_all.shape[1] * 4) if amem_emb_all.ndim == 2 else 0
            amem_text_bytes = 0
            for m in amem_memories_all[:n_turns]:
                # MemoryNote fields used in retrieval context string
                amem_text_bytes += _utf8_len(getattr(m, "timestamp", ""))
                amem_text_bytes += _utf8_len(getattr(m, "content", ""))
                amem_text_bytes += _utf8_len(getattr(m, "context", ""))
                amem_text_bytes += _utf8_len(getattr(m, "keywords", None))
                amem_text_bytes += _utf8_len(getattr(m, "tags", None))
                amem_text_bytes += _utf8_len(getattr(m, "links", None))

            amem_storage_mb = (amem_text_bytes + amem_turn_emb_bytes) / 1_000_000.0
            amem_retrieval_ms = _bench_topk_search_ms(amem_emb_norm_all[:n_turns], k=50) if n_turns else 0.0

            rows.append(
                {
                    "sample_id": sample_id,
                    "prefix_turns": n_turns,
                    "higmem_turn_count": n_turns,
                    "higmem_event_count_est": len(event_ids_prefix),
                    "higmem_storage_mb_est": higmem_storage_mb,
                    "higmem_vector_retrieval_ms_est": higmem_retrieval_ms,
                    "amem_turn_count": n_turns,
                    "amem_storage_mb_est": amem_storage_mb,
                    "amem_vector_retrieval_ms_est": amem_retrieval_ms,
                }
            )

    # Aggregate across samples
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        n = r["prefix_turns"]
        for k, v in r.items():
            if k in {"sample_id", "prefix_turns"}:
                continue
            agg[n][k].append(v)

    out_rows = []
    for n in prefix_sizes:
        if n not in agg:
            continue
        d = {"prefix_turns": n, "num_samples": len(agg[n].get("amem_storage_mb_est", []))}
        for k, vals in agg[n].items():
            if not vals:
                continue
            # Average
            d[k] = float(np.mean(vals))
        out_rows.append(d)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write CSV
    fieldnames = sorted(set().union(*(r.keys() for r in out_rows)))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"Wrote: {out_path}")
    print("Averages:")
    for r in out_rows:
        print(r)

    # Optional synthetic scaling (vector-only) for larger N.
    if args.simulate_sizes.strip():
        sim_sizes = [int(x.strip()) for x in args.simulate_sizes.split(",") if x.strip()]
        sim_sizes = sorted(set(sim_sizes))
        dim = int(args.simulate_embed_dim)
        ratio = float(args.simulate_event_ratio)
        rng = np.random.default_rng(0)

        sim_rows = []
        for n in sim_sizes:
            n = int(n)
            if n <= 0:
                continue
            # A-Mem: N turn embeddings
            E = rng.standard_normal((n, dim), dtype=np.float32)
            E = _normalize_rows(E)
            reps = 10 if n >= 100_000 else 50
            amem_ms = _bench_topk_search_ms(E, k=50, reps=reps, seed=0)
            amem_mb = (n * dim * 4) / 1_000_000.0

            # HiGMem: N turns + ~ratio*N events
            m = max(1, int(n * ratio))
            Ev = rng.standard_normal((m, dim), dtype=np.float32)
            Ev = _normalize_rows(Ev)
            hig_turn_ms = _bench_topk_search_ms(E, k=10, reps=reps, seed=1)
            hig_event_ms = _bench_topk_search_ms(Ev, k=int(args.higmem_k_event), reps=reps, seed=2)
            hig_ms = hig_turn_ms + hig_event_ms
            hig_mb = ((n + m) * dim * 4) / 1_000_000.0

            sim_rows.append(
                {
                    "turns": n,
                    "events_est": m,
                    "amem_storage_mb_embeddings_only": amem_mb,
                    "amem_vector_retrieval_ms_est": amem_ms,
                    "higmem_storage_mb_embeddings_only": hig_mb,
                    "higmem_vector_retrieval_ms_est": hig_ms,
                }
            )

        sim_out = out_path.parent / "exp5_scaling_simulated.csv"
        if sim_rows:
            fieldnames = list(sim_rows[0].keys())
            with open(sim_out, "w", encoding="utf-8", newline="") as f:
                import csv

                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in sim_rows:
                    w.writerow(r)
            print(f"Wrote: {sim_out}")
            for r in sim_rows:
                print(r)


if __name__ == "__main__":
    main()
