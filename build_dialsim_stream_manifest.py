import argparse
import gc
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dialsim_dataset import (
    DEFAULT_DIALSIM_SHOWS,
    load_pickle_from_source,
    parse_script_to_turns,
    iter_easy_questions,
    iter_hard_questions,
)


_DATE_SUFFIX_RE = re.compile(r"(\d+)(st|nd|rd|th)", re.IGNORECASE)
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_id(s: str) -> str:
    s = str(s or "").strip()
    s = _SAFE_ID_RE.sub("_", s)
    return s.strip("_") or "unknown"


def _parse_date(date_str: str) -> Optional[datetime]:
    s = str(date_str or "").strip()
    if not s:
        return None
    s = _DATE_SUFFIX_RE.sub(r"\1", s)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _allocate_equal(total: int, keys: List[str]) -> Dict[str, int]:
    """Evenly split total across keys (deterministic remainder assignment by key order)."""
    if total < 0:
        raise ValueError("total must be >= 0")
    if not keys:
        return {}
    base = total // len(keys)
    rem = total % len(keys)
    out = {}
    for i, k in enumerate(keys):
        out[k] = base + (1 if i < rem else 0)
    return out


def _largest_remainder_targets(weights: Dict[str, float], total: int) -> Dict[str, int]:
    """Convert weights (sum~=1) into integer targets that sum to total."""
    if total < 0:
        raise ValueError("total must be >= 0")
    raw = {k: float(w) * total for k, w in weights.items()}
    floors = {k: int(raw[k]) for k in raw}
    remain = total - sum(floors.values())
    if remain <= 0:
        return floors
    remainders = sorted(((raw[k] - floors[k], k) for k in raw), reverse=True)
    out = dict(floors)
    for _, k in remainders:
        if remain <= 0:
            break
        out[k] += 1
        remain -= 1
    return out


def _bucket_len(bucket: Any) -> int:
    if isinstance(bucket, dict):
        return len(bucket)
    if isinstance(bucket, list):
        return len(bucket)
    return 0


def _collect_global_bucket_counts(dialsim_source: str, shows: List[str], third_party_dir: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for show in shows:
        member = f"{show}_dialsim.pickle"
        print(f"[counts] loading {member} ...")
        show_data = load_pickle_from_source(dialsim_source, member, third_party_dir=third_party_dir)
        for _ep_name, ep in show_data.items():
            if not isinstance(ep, dict):
                continue
            for _scene_id, scene in ep.items():
                if not isinstance(scene, dict):
                    continue
                easy_q = scene.get("easy_q") or {}
                if isinstance(easy_q, dict):
                    for bname, bucket in easy_q.items():
                        counts[bname] = counts.get(bname, 0) + _bucket_len(bucket)
                hard_q = scene.get("hard_q") or {}
                if isinstance(hard_q, dict):
                    for bname, bucket in hard_q.items():
                        counts[bname] = counts.get(bname, 0) + _bucket_len(bucket)

        # free aggressively (DialSim pickles are large)
        del show_data
        gc.collect()
    return counts


def _scene_sort_key(date_str: str, episode: str, scene_id: int) -> Tuple[int, str, str, int]:
    dt = _parse_date(date_str)
    if dt is None:
        # Put unparsable dates at the end, but keep deterministic ordering.
        return (1, "9999-12-31", str(episode), int(scene_id))
    return (0, dt.strftime("%Y-%m-%d"), str(episode), int(scene_id))


def _iter_scene_questions(scene_item: dict, show: str, episode: str, scene_id: int) -> List[dict]:
    # Enumerate *all* buckets present in this scene (not just oracle fan buckets).
    easy_q = scene_item.get("easy_q") or {}
    hard_q = scene_item.get("hard_q") or {}
    easy_types = list(easy_q.keys()) if isinstance(easy_q, dict) else []
    hard_types = list(hard_q.keys()) if isinstance(hard_q, dict) else []

    out: List[dict] = []
    for q in iter_easy_questions(show, episode, scene_id, scene_item, include_q_types=easy_types):
        out.append(
            {
                "split": "easy",
                "q_type": q.q_type,
                "q_id": q.q_id,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "show": q.show,
                "episode": q.episode,
                "scene_id": q.scene_id,
                "date": q.date,
            }
        )
    for q in iter_hard_questions(show, episode, scene_id, scene_item, include_q_types=hard_types):
        out.append(
            {
                "split": "hard",
                "q_type": q.q_type,
                "q_id": q.q_id,
                "question": q.question,
                "options": q.options,
                "answer": q.answer,
                "show": q.show,
                "episode": q.episode,
                "scene_id": q.scene_id,
                "date": q.date,
            }
        )
    return out


def _reservoir_add(rng: random.Random, reservoir: List[dict], target_k: int, seen: int, item: dict) -> int:
    """Reservoir sampling update; returns new seen count."""
    seen += 1
    if target_k <= 0:
        return seen
    if len(reservoir) < target_k:
        reservoir.append(item)
        return seen
    j = rng.randrange(seen)
    if j < target_k:
        reservoir[j] = item
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained DialSim streaming eval manifest (turn-level + scene-end QA).")
    default_source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "HiGMem_Other", "dialsim_v1.1.zip"))
    parser.add_argument("--dialsim_source", type=str, default=default_source, help="DialSim v1.1 zip or extracted dir.")
    parser.add_argument("--shows", type=str, default=",".join(DEFAULT_DIALSIM_SHOWS), help="Comma-separated shows.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for manifest sampling.")
    parser.add_argument("--turns_total", type=int, default=7000, help="Total turns budget across all shows.")
    parser.add_argument("--questions_total", type=int, default=3000, help="Total questions budget across all shows.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).parent / "dialsim_manifests" / "dialsim_v1.1_stream_t7000_q3000_seed0.json"),
        help="Where to write the manifest JSON.",
    )
    args = parser.parse_args()

    shows = [s.strip() for s in args.shows.split(",") if s.strip()]
    if not shows:
        raise ValueError("No shows specified.")

    third_party_dir = os.path.join(os.path.dirname(__file__), "third_party", "zipfile_deflate64")

    # Phase 1: global bucket distribution over full DialSim v1.1 (counts only; fast once loaded).
    bucket_counts_full = _collect_global_bucket_counts(args.dialsim_source, shows, third_party_dir)
    total_questions_full = int(sum(bucket_counts_full.values()))
    if total_questions_full <= 0:
        raise RuntimeError("No questions found in DialSim source (bucket counts sum to 0).")
    bucket_weights_full = {k: bucket_counts_full[k] / total_questions_full for k in sorted(bucket_counts_full.keys())}

    # Budgets: equal per show by default (deterministic remainder assignment by show order).
    turns_target_by_show = _allocate_equal(int(args.turns_total), shows)
    questions_target_by_show = _allocate_equal(int(args.questions_total), shows)

    print("[budgets] turns_target_by_show:", turns_target_by_show)
    print("[budgets] questions_target_by_show:", questions_target_by_show)

    manifest: Dict[str, Any] = {
        "dialsim_version": "v1.1",
        "seed": int(args.seed),
        "targets": {
            "turns_total": int(args.turns_total),
            "questions_total": int(args.questions_total),
            "turns_by_show": turns_target_by_show,
            "questions_by_show": questions_target_by_show,
        },
        "sampling": {
            "bucket_counts_full": bucket_counts_full,
            "bucket_weights_full": bucket_weights_full,
            "note": "Bucket weights are computed over the full DialSim v1.1 dataset (all shows).",
        },
        "shows": {},
    }

    # Phase 2: per-show scene selection + per-bucket reservoir sampling over eligible scenes.
    for show_idx, show in enumerate(shows):
        rng = random.Random(int(args.seed) + 1009 * show_idx)
        member = f"{show}_dialsim.pickle"
        print(f"[manifest] loading {member} ...")
        show_data = load_pickle_from_source(args.dialsim_source, member, third_party_dir=third_party_dir)

        # Pre-list all scenes so we can sort chronologically.
        scene_refs: List[Tuple[str, int, str]] = []
        for ep_name, ep in show_data.items():
            if not isinstance(ep, dict):
                continue
            for scene_id, scene_item in ep.items():
                if not isinstance(scene_item, dict):
                    continue
                date_str = str(scene_item.get("date", "") or "")
                scene_refs.append((str(ep_name), int(scene_id), date_str))

        scene_refs.sort(key=lambda x: _scene_sort_key(x[2], x[0], x[1]))

        turns_target = int(turns_target_by_show.get(show, 0))
        questions_target = int(questions_target_by_show.get(show, 0))

        # Bucket targets for this show (same weights as full dataset).
        bucket_targets = _largest_remainder_targets(bucket_weights_full, questions_target)
        # Drop buckets that are globally empty (shouldn't happen) or have 0 target.
        bucket_targets = {k: int(v) for k, v in bucket_targets.items() if int(v) > 0 and bucket_counts_full.get(k, 0) > 0}

        # Reservoirs per bucket.
        bucket_seen = {k: 0 for k in bucket_targets}
        bucket_reservoir: Dict[str, List[dict]] = {k: [] for k in bucket_targets}

        scenes_out: List[dict] = []
        turns_so_far = 0
        full_scenes_for_sampling: List[Tuple[str, int, dict]] = []

        for ep_name, scene_id, date_str in scene_refs:
            if turns_so_far >= turns_target:
                break
            scene_item = show_data.get(ep_name, {}).get(scene_id, {})
            if not isinstance(scene_item, dict):
                continue
            script = str(scene_item.get("script", "") or "")
            parsed_turns = parse_script_to_turns(script)
            if not parsed_turns:
                continue

            remaining = turns_target - turns_so_far
            take_n = min(len(parsed_turns), remaining)

            safe_ep = _safe_id(ep_name)
            scene_uid = f"{show}|{safe_ep}|scene{int(scene_id)}"
            turns_list: List[dict] = []
            for i, (spk, txt) in enumerate(parsed_turns[:take_n]):
                turn_id = f"{scene_uid}|turn{i:04d}"
                ts = f"{date_str}|{safe_ep}|scene{int(scene_id)}|{i:04d}"
                turns_list.append(
                    {
                        "turn_id": turn_id,
                        "speaker": str(spk),
                        "text": str(txt),
                        "timestamp": ts,
                    }
                )

            scene_entry = {
                "show": show,
                "episode": ep_name,
                "scene_id": int(scene_id),
                "date": date_str,
                "scene_uid": scene_uid,
                "is_partial": bool(take_n < len(parsed_turns)),
                "num_turns_total_in_scene": int(len(parsed_turns)),
                "turns": turns_list,
                "questions": [],  # filled later
            }
            scenes_out.append(scene_entry)
            turns_so_far += take_n

            # Only full scenes are eligible for scene-end QA.
            if take_n == len(parsed_turns):
                full_scenes_for_sampling.append((ep_name, int(scene_id), scene_item))
            else:
                break  # stop at partial last scene

        # First pass: reservoir per bucket on eligible scenes.
        for ep_name, scene_id, scene_item in full_scenes_for_sampling:
            for q in _iter_scene_questions(scene_item, show, ep_name, scene_id):
                b = q.get("q_type", "")
                if b not in bucket_targets:
                    continue
                # Basic sanity filtering (avoid empty fields that would break prompts).
                if not q.get("question") or not q.get("options") or not q.get("answer"):
                    continue
                bucket_seen[b] = _reservoir_add(rng, bucket_reservoir[b], int(bucket_targets[b]), int(bucket_seen[b]), q)

        selected: List[dict] = []
        selected_ids: set = set()
        for b in sorted(bucket_reservoir.keys()):
            for q in bucket_reservoir[b]:
                q_uid = f"{q['episode']}|{q['scene_id']}|{q['q_type']}|{q['q_id']}"
                if q_uid in selected_ids:
                    continue
                selected_ids.add(q_uid)
                selected.append(q)

        # Fill any deficit (rare buckets may be under-populated in the prefix scenes).
        missing = questions_target - len(selected)
        if missing > 0:
            fill_seen = 0
            fill_res: List[dict] = []
            for ep_name, scene_id, scene_item in full_scenes_for_sampling:
                for q in _iter_scene_questions(scene_item, show, ep_name, scene_id):
                    q_uid = f"{q['episode']}|{q['scene_id']}|{q['q_type']}|{q['q_id']}"
                    if q_uid in selected_ids:
                        continue
                    if not q.get("question") or not q.get("options") or not q.get("answer"):
                        continue
                    fill_seen = _reservoir_add(rng, fill_res, missing, fill_seen, q)
            for q in fill_res:
                q_uid = f"{q['episode']}|{q['scene_id']}|{q['q_type']}|{q['q_id']}"
                if q_uid in selected_ids:
                    continue
                selected_ids.add(q_uid)
                selected.append(q)

        # If we somehow overshot due to dedup/edge cases, trim deterministically.
        if len(selected) > questions_target:
            selected = sorted(selected, key=lambda x: (x["date"], x["episode"], int(x["scene_id"]), x["q_type"], x["q_id"]))[:questions_target]

        if len(selected) != questions_target:
            raise RuntimeError(f"Show {show}: expected {questions_target} questions, got {len(selected)}.")

        # Attach questions back to scenes (scene-end insertion).
        q_by_scene: Dict[str, List[dict]] = {}
        for q in selected:
            key = f"{q['episode']}|{int(q['scene_id'])}"
            q_by_scene.setdefault(key, []).append(q)
        for qs in q_by_scene.values():
            qs.sort(key=lambda x: (x["q_type"], x["q_id"]))

        for scene in scenes_out:
            key = f"{scene['episode']}|{int(scene['scene_id'])}"
            scene["questions"] = q_by_scene.get(key, [])

        # Summaries for this show.
        selected_bucket_counts: Dict[str, int] = {}
        idk = "I don't know. (None of the above)"
        idk_cnt = 0
        for q in selected:
            selected_bucket_counts[q["q_type"]] = selected_bucket_counts.get(q["q_type"], 0) + 1
            if str(q.get("answer", "")).strip() == idk:
                idk_cnt += 1

        manifest["shows"][show] = {
            "turns_target": turns_target,
            "questions_target": questions_target,
            "turns_in_manifest": turns_so_far,
            "questions_in_manifest": len(selected),
            "scenes": scenes_out,
            "selected_bucket_counts": selected_bucket_counts,
            "selected_idk_fraction": (idk_cnt / len(selected)) if selected else 0.0,
        }

        print(f"[manifest] {show}: turns={turns_so_far}/{turns_target} scenes={len(scenes_out)} questions={len(selected)}/{questions_target} idk%={100.0 * (idk_cnt / len(selected)):.2f}")

        # free aggressively
        del show_data
        gc.collect()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote manifest: {out_path}")


if __name__ == "__main__":
    main()

