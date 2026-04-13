# Open-Source Model (vLLM / OpenAI-Compatible) Setup Notes

Goal: run **HiGMem on locomo10** while calling an **OpenAI-compatible** inference server (e.g., vLLM) from this machine.

## What Was Changed

- `H-Mem/memory_layer.py` `OpenAIController.get_completion()` now:
  - honors `OPENAI_MAX_TOKENS` (optional) to avoid server-side max token rejections
  - retries **without** `response_format` if the server rejects `response_format/json_schema`
- `H-Mem/fphm_core.py` `_get_llm_json_response()` now tries to salvage JSON if the model wraps JSON in extra text.

These changes should not affect normal GPT runs (they only activate on errors / overrides).

## Environment Variables

Set these before running:

- `OPENAI_API_BASE`: `http://<server>:<port>/v1`
- `OPENAI_API_KEY`: whatever your server expects; if no auth, `EMPTY` is usually fine
- Optional: `OPENAI_MAX_TOKENS`: e.g. `2048` (only needed if the server rejects large defaults)

## Run locomo10 (HiGMem)

```powershell
cd D:\PycharmProjects\H-Mem
.\.venv\Scripts\python.exe run_fphm_evaluation.py `
  --dataset data\locomo10.json `
  --backend openai `
  --model <your-vllm-model-name> `
  --api_base $env:OPENAI_API_BASE `
  --api_key $env:OPENAI_API_KEY `
  --num-workers 10 `
  --parallel-backend thread
```

If the server is fast and stable, higher concurrency is fine; the local machine mainly does retrieval + logging.

## Run locomo10 (A-Mem baseline)

`AgenticMemory/test_advanced.py` supports the same `--api_base/--api_key` flags and thread parallelism.

```powershell
cd D:\PycharmProjects\AgenticMemory
python test_advanced.py `
  --dataset data\locomo10.json `
  --backend openai `
  --model <your-vllm-model-name> `
  --api_base $env:OPENAI_API_BASE `
  --api_key $env:OPENAI_API_KEY `
  --ratio 1 `
  --retrieve_k 50 `
  --num_workers 10 `
  --parallel_backend thread
```
