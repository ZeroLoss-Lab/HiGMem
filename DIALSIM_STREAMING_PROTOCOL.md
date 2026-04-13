# DialSim v1.1 Streaming Eval (1w Budget)

This repo now supports a **streaming / real-time** DialSim evaluation protocol that matches what we discussed:

- **Turn-level** memory construction (no scene as a single blob)
- **Scene-end** question insertion (answer only after all turns in the scene are visible)
- **Per-show timelines** (friends / bigbang / theoffice are not mixed)
- **Weighted sampling** to match the dataset's global bucket distribution
- A-Mem and HiGMem run the **same manifest** for a fair comparison.

## Protocol Summary

1) Streaming memory construction (turn-level)
- For each show independently:
  - Sort scenes by `(date, episode, scene_id)`
  - For each scene, stream turns and call `add_turn` (HiGMem) / `add_note` (A-Mem)

2) Question insertion (scene-end)
- After finishing *all* turns of a scene, answer the sampled questions from that scene.

3) Budget split
- Default: `turns_total=7000`, `questions_total=3000` (turns + questions <= 10,000)
- Shows are split evenly (deterministic remainder assignment).

4) Sampling
- A self-contained **manifest JSON** is generated first.
- The manifest includes:
  - The exact turns (speaker/text/timestamp/turn_id)
  - The exact questions (bucket/q_id/question/options/answer)
- Question buckets are sampled according to **full DialSim v1.1 global bucket weights**.

## Files / Entry Points

- Build manifest (self-contained; avoids re-loading huge pickle QA pools during eval):
  - `H-Mem/build_dialsim_stream_manifest.py`
- Run HiGMem streaming eval:
  - `H-Mem/run_dialsim_streaming_eval.py`
- Run A-Mem streaming eval:
  - `AgenticMemory/run_dialsim_streaming_eval.py`
- Analyze predictions (Table-2 style metrics + retrieval proxy stats):
  - `H-Mem/analyze_dialsim_results.py`

## Commands (Typical)

Build a 1w manifest (7000 turns + 3000 questions):

```powershell
cd D:\PycharmProjects\H-Mem
.\.venv\Scripts\python.exe build_dialsim_stream_manifest.py `
  --seed 42 `
  --turns_total 7000 `
  --questions_total 3000 `
  --output dialsim_manifests\dialsim_v1.1_stream_t7000_q3000_seed42.json
```

Run HiGMem:

```powershell
cd D:\PycharmProjects\H-Mem
.\.venv\Scripts\python.exe run_dialsim_streaming_eval.py `
  --manifest dialsim_manifests\dialsim_v1.1_stream_t7000_q3000_seed42.json `
  --model gpt-4o-mini --backend openai --api_base $env:OPENAI_API_BASE
```

Run A-Mem:

```powershell
cd D:\PycharmProjects\AgenticMemory
.\.venv\Scripts\python.exe run_dialsim_streaming_eval.py `
  --manifest ..\H-Mem\dialsim_manifests\dialsim_v1.1_stream_t7000_q3000_seed42.json `
  --model gpt-4o-mini --backend openai --api_base $env:OPENAI_API_BASE
```

Analyze:

```powershell
cd D:\PycharmProjects\H-Mem
.\.venv\Scripts\python.exe analyze_dialsim_results.py `
  --input dialsim_runs\<RUN>\predictions.jsonl `
  --scale_100
```

## Notes

- `parse_script_to_turns()` speaker parsing regex was fixed to correctly parse lines like `Monica: ...`.
- The final multiple-choice QA prompt is duplicated verbatim in both streaming eval scripts to keep it identical.
- If you use an OpenAI-compatible open-source server (vLLM), you can set `OPENAI_MAX_TOKENS` to avoid server-side max token rejections.

