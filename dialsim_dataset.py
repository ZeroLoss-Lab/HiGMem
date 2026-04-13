import os
import re
import sys
import pickle
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


DEFAULT_DIALSIM_SHOWS = ("friends", "bigbang", "theoffice")
# The "fan" (easy) split is what the provided *_oracle_fan.pickle covers:
# it includes ans_w_time, ans_wo_time, and dont_know_unans_time; other easy buckets exist
# but are empty in the oracle files.
DEFAULT_EASY_Q_TYPES = ("ans_w_time", "ans_wo_time", "dont_know_unans_time")
# Hard question buckets seen in DialSim v1.1 pickles (temporal reasoning variants).
# Note: We don't assume every bucket is present in every scene.
DEFAULT_HARD_Q_TYPES = (
    "past",
    "cur",
    "fu",
    "past_past",
    "past_cur",
    "cur_past",
    "cur_cur",
    "past_fu",
    "fu_past",
    "fu_cur",
    "cur_fu",
    "fu_fu",
)


def _try_enable_deflate64_zip_support(third_party_dir: Optional[str]) -> None:
    """Enable reading Deflate64-compressed ZIPs (DialSim zips) if possible.

    The provided DialSim zips are compressed with Deflate64, which Python's stdlib `zipfile`
    cannot read. We vendor `zipfile-deflate64` into each project under third_party/ so this
    works on Windows without installing system-wide dependencies.
    """
    if not third_party_dir:
        return
    if third_party_dir not in sys.path:
        sys.path.insert(0, third_party_dir)


def _open_zipfile_deflate64(zip_path: str, third_party_dir: Optional[str]):
    _try_enable_deflate64_zip_support(third_party_dir)
    try:
        from zipfile_deflate64 import ZipFile  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "DialSim zip uses Deflate64 compression. Install/ship 'zipfile-deflate64' or "
            "extract the zip with 7zip and pass a directory path instead."
        ) from e
    return ZipFile(zip_path)


def load_pickle_from_source(source_path: str, member_name: str, third_party_dir: Optional[str] = None):
    """Load a pickle either from a Deflate64-compressed zip or from an extracted directory."""
    source_path = os.path.abspath(source_path)
    if os.path.isdir(source_path):
        pkl_path = os.path.join(source_path, member_name)
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    if source_path.lower().endswith(".zip"):
        with _open_zipfile_deflate64(source_path, third_party_dir) as z:
            with z.open(member_name, "r") as f:
                return pickle.load(f)

    raise ValueError(f"Unsupported DialSim source path: {source_path} (expected .zip or directory)")


# NOTE: this is standard whitespace matching after the speaker prefix, e.g. "Michael: ...".
_SPEAKER_RE = re.compile(r"^([^:]{1,60}):\s*(.*)$")


def parse_script_to_turns(script_text: str) -> List[Tuple[str, str]]:
    """Parse a DialSim `script` string into (speaker, text) turns.

    Lines are typically like "Michael: ...". If a line does not match, we keep it under
    a 'Narrator' pseudo-speaker so content is not silently dropped.
    """
    turns: List[Tuple[str, str]] = []
    for raw in (script_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            speaker = m.group(1).strip()
            text = m.group(2).strip()
        else:
            speaker = "Narrator"
            text = line
        if text:
            turns.append((speaker, text))
    return turns


@dataclass(frozen=True)
class DialSimQuestion:
    show: str
    episode: str
    scene_id: int
    date: str
    q_type: str
    q_id: str
    question: str
    options: List[str]
    answer: str


def iter_easy_questions(
    show: str,
    episode: str,
    scene_id: int,
    scene_item: dict,
    *,
    include_q_types: Iterable[str] = DEFAULT_EASY_Q_TYPES,
) -> Iterator[DialSimQuestion]:
    """Yield easy (multiple-choice) questions from a scene item."""
    date = str(scene_item.get("date", "") or "")
    easy_q = scene_item.get("easy_q", {}) or {}
    if not isinstance(easy_q, dict):
        return

    for q_type in include_q_types:
        bucket = easy_q.get(q_type, {}) or {}
        if not isinstance(bucket, dict):
            continue
        for qid_raw, qobj in bucket.items():
            if not isinstance(qobj, dict):
                continue
            questions = qobj.get("questions", {}) or {}
            if isinstance(questions, dict):
                q_text = questions.get("default") or next(iter(questions.values()), "")
            else:
                q_text = ""
            q_text = str(q_text or "").strip()
            options = qobj.get("options", []) or []
            if not isinstance(options, list):
                options = []
            options = [str(x) for x in options]
            answer = str(qobj.get("answer", "") or "").strip()
            yield DialSimQuestion(
                show=show,
                episode=episode,
                scene_id=int(scene_id),
                date=date,
                q_type=str(q_type),
                q_id=str(qid_raw),
                question=q_text,
                options=options,
                answer=answer,
            )


def iter_hard_questions(
    show: str,
    episode: str,
    scene_id: int,
    scene_item: dict,
    *,
    include_q_types: Iterable[str] = DEFAULT_HARD_Q_TYPES,
) -> Iterator[DialSimQuestion]:
    """Yield hard (multiple-choice) questions from a scene item.

    In DialSim v1.1, `hard_q` buckets are typically lists (not dicts), each element being a dict
    with keys like: {questions, options, answer}.
    """
    date = str(scene_item.get("date", "") or "")
    hard_q = scene_item.get("hard_q", {}) or {}
    if not isinstance(hard_q, dict):
        return

    for q_type in include_q_types:
        bucket = hard_q.get(q_type, None)
        if bucket is None:
            continue

        # Most commonly: list[dict]
        if isinstance(bucket, list):
            for idx, qobj in enumerate(bucket):
                if not isinstance(qobj, dict):
                    continue
                questions = qobj.get("questions", {}) or {}
                if isinstance(questions, dict):
                    q_text = questions.get("default") or next(iter(questions.values()), "")
                else:
                    q_text = ""
                q_text = str(q_text or "").strip()
                options = qobj.get("options", []) or []
                if not isinstance(options, list):
                    options = []
                options = [str(x) for x in options]
                answer = str(qobj.get("answer", "") or "").strip()
                yield DialSimQuestion(
                    show=show,
                    episode=episode,
                    scene_id=int(scene_id),
                    date=date,
                    q_type=str(q_type),
                    q_id=str(idx),
                    question=q_text,
                    options=options,
                    answer=answer,
                )
            continue

        # Fallback: dict[qid -> dict]
        if isinstance(bucket, dict):
            for qid_raw, qobj in bucket.items():
                if not isinstance(qobj, dict):
                    continue
                questions = qobj.get("questions", {}) or {}
                if isinstance(questions, dict):
                    q_text = questions.get("default") or next(iter(questions.values()), "")
                else:
                    q_text = ""
                q_text = str(q_text or "").strip()
                options = qobj.get("options", []) or []
                if not isinstance(options, list):
                    options = []
                options = [str(x) for x in options]
                answer = str(qobj.get("answer", "") or "").strip()
                yield DialSimQuestion(
                    show=show,
                    episode=episode,
                    scene_id=int(scene_id),
                    date=date,
                    q_type=str(q_type),
                    q_id=str(qid_raw),
                    question=q_text,
                    options=options,
                    answer=answer,
                )
