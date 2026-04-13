# HiGMem / A-Mem 补实验执行记录

本文件用于持久记录：运行命令、输出目录、关键表格与结论，避免上下文压缩导致遗漏。

## 约束与约定（2026-02-18）
- 数据集：LoCoMo `locomo10`
- 总预算：<= 300 USD（尽量复用已有 run/caches，避免重复跑）
- 不修改无关工程：不动 `EducationMVP*` 等目录
- 允许修改 `AgenticMemory` 用于统计/日志，但不得改变影响 F1 的行为
- 若修改了“记忆构筑 / 构筑缓存格式 / 构筑过程依赖的结构体”，必须先清理对应缓存（否则脚本会检测缓存并跳过构筑，直接进入 QA，导致“修改未生效”的假象）。
- **重要（2026-02-21）**：论文设定 `k_event=10`，但当时代码默认值被误写为 `7`。因此本文件中 **2026-02-21 之前**的 HiGMem 指标均应视为 *k_event=7 legacy*，在补实验/对齐论文口径前需要用 `k_event=10` 重新跑（尤其会影响 Exp3/4/5 以及所有 locomo10 主方法 runs）。
- **重要（2026-02-22）**：DialSim 仅用于 **Exp3**。其余 Exp1/2/4/5 的分析与写作只基于 `locomo10`（流程/代码与 locomo10 保持一致即可，避免在非 Exp3 里引入 DialSim 口径差异）。

## 实验清单
- Exp1：Flat Retrieval + LLM Selection baseline（HiGMem 去 Event 层）
- Exp2：A-Mem Recall@fixedK（K 以回答 LLM 看到的 turns 数量为准，含邻居扩展）
- Exp4：End-to-End Token Cost（检索 + 生成）对比 HiGMem vs A-Mem
- Exp5：Scaling Analysis（存储开销、检索延迟、复杂度）
- Exp3：DialSim（低优先级，视时间/预算再做）

## 单独报告文件（方便发给“不看代码的解释者”）
- Exp1：`H-Mem/analysis_results/exp1_locomo10_flat_baseline_report.md`
- Exp2：`H-Mem/analysis_results/exp2_amem_recall_fixedk_report.md`
- Exp4：`H-Mem/analysis_results/exp4_token_cost_report.md`
- Exp5：`H-Mem/analysis_results/exp5_scaling_report.md`
- Exp3：`H-Mem/analysis_results/exp3_dialsim_report.md`

## 一页结论（locomo10，可直接复制到 rebuttal / 需求对接）
- 数据集：`locomo10`（10 samples；共 1986 QAs，其中 4 题 evidence 为空；所有 recall/precision 统计忽略这 4 题 → n=1982）
- HiGMem（主方法，**论文口径**；config=`gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync`）：
  - F1：Overall=0.4808（Cat1=0.3163, Cat2=0.3531, Cat3=0.1480, Cat4=0.4896, Cat5=0.7318）
    - 汇总：`H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync_20260222_003900/aggregated_results.json`
  - 检索质量（最终给回答 LLM 的 turns；按 evidence 非空题，n=1982）：Precision=0.1336，Recall=0.7469，avg K=12.1877
    - 汇总：`H-Mem/analysis_results/recall_and_cost_summary_BEST_49.csv`（Overall 行）
  - Token/Time（QA 平均；按 n=1982）：avg prompt tokens=15110.82；avg answer prompt=1177.19；avg time=4.71s
- Exp1 Flat baseline（config=`gpt-4o-mini_no_event_no_profile_kturn100_sync`，top-100 turns + LLM filtering）：
  - F1：Overall=0.4673（Cat1=0.2921, Cat2=0.3767, Cat3=0.1158, Cat4=0.4607, Cat5=0.7312）
  - 检索质量（最终 turns）：Precision=0.0656，Recall=0.7030，avg K=22.7053
  - Token（QA 平均）：avg prompt tokens=8556.99；其中 answer prompt=1832.08
- Exp2 A-Mem Recall@fixedK（按“回答 LLM 实际看到的 expanded turns”计 K）：full(avg K≈99.84) Recall=0.7502；截断到 K=8/16/32 时 Recall 明显低于 HiGMem 自然输出（见 Exp2 表）
- Exp4 Token Cost：HiGMem 把“最终回答阶段输入 tokens”从 A-Mem 的 ~12.8k/题 降到 ~1.18k/题（≈10.8×）；在“retrieval=mini, answer=gpt-5（仅算 input）”的 hybrid 设定下，A-Mem ≈ $15.86，而 HiGMem ≈ $3.53（见 `H-Mem/analysis_results/exp4_token_cost_report.json`）
- Exp5 Scaling：两者存储均随 turns 线性增长；HiGMem 因额外 event 文本/向量带来约 +8.5% 的存储与轻微的 vector top-k 计算开销（embeddings-only 模拟到 1M turns：HiGMem ≈ 53.04ms vs A-Mem ≈ 47.70ms）
- Exp3 DialSim：只用于 Exp3（其余实验不分析 DialSim；见本文 Exp3 章节）

## 已发现的已有产物（未新增运行）
- HiGMem 既有 runs：`H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_sync_20260107_155735` 等
- A-Mem caches：`AgenticMemory/cached_memories_advanced_openai_gpt-4o-mini/*`（sample 0-9 全量）
- A-Mem 既有结果文件：`AgenticMemory/results-ori-ratio0.1-k50-lconai-4omini-1-3.json`（仅 0.1 ratio，且会被旧脚本覆盖，需后续改名输出）

## 2026-02-19 进程清理记录
- 用户提示“梯子未关可能导致链接不稳定”，已确认当前环境无 `*_PROXY` 环境变量。
- 发现上一次被打断的 HiGMem Exp1 尝试残留 4 个 `run_fphm_evaluation.py` Python 进程（PID: 8732, 24172, 22464, 10328），已强制结束。
- 因中断产生的未完成输出目录（仅部分 sample 日志、无 aggregated_results / final checkpoint）：
  - `H-Mem/fphm_runs/gpt-4o-mini_no_event_no_profile_sync_20260219_081500`
  - `H-Mem/fphm_runs/gpt-4o-mini_no_event_no_profile_sync_20260219_082606`

## 2026-02-19 GPU 显存异常排查（用户反馈“10 个 python 进程 + 显存 5.8GB+1.4GB”）
- `nvidia-smi` 显示 GPU0 占用约 6688MiB，且出现 10 个 `python.exe` 进程。
- 通过 `Get-CimInstance Win32_Process` 反查命令行，这 10 个进程均为
  `python.exe -c "from multiprocessing.spawn import spawn_main; ... parent_pid=24172/10328" --multiprocessing-fork`
  的 **multiprocessing 子进程**，父进程是之前被中止的 HiGMem 跑实验进程。
- 已强制结束这些 PID（6672/6748/10124/12312/13180/13732/14664/21340/22200/24100），显存恢复到 ~379MiB。

## 2026-02-19 并行/显存修复（为“边跑实验边打游戏”）
- 结论：**线程**在同一进程内共享 CUDA context/VRAM；**进程**会重复创建 CUDA context + 重复加载模型，显存线性膨胀，且中断时可能残留子进程。
- HiGMem：`H-Mem/run_fphm_evaluation.py` 新增 `--parallel-backend thread|process`（默认 thread），避免多进程显存爆炸。
- HiGMem：`H-Mem/memory_layer.py` 对 `SentenceTransformer` 做了 per-process cache，避免 turn/event/profile 三个 retriever 重复加载同一模型。
- A-Mem：`AgenticMemory/test_advanced.py` 新增 `--parallel_backend thread|process`（默认 thread），默认不再启动大量 python 子进程。
- A-Mem：`AgenticMemory/memory_layer.py` 对 `SentenceTransformer` 做了 per-process cache，便于线程并行时多 sample 复用同一份模型权重。
- A-Mem：`AgenticMemory/test_advanced.py` 移除未使用的全局 `SentenceTransformer('all-MiniLM-L6-v2')` 载入（原先仅占显存，无任何用途）。

## 2026-02-19 运行时显存估算（按“正常复现设置”加载本地模型）
在两套 venv 中各跑了一次“加载检索器 ST + 跑一次 calculate_metrics 以触发 BERTScore/SBERT 模型加载”的小脚本，`nvidia-smi` 观测到的总显存占用一致：
- 初始（仅桌面/常驻）：~544 MiB
- 仅加载 `SentenceTransformer(all-MiniLM-L6-v2)`：~739 MiB
- 触发 `calculate_metrics()`（加载 BERTScore 的 `roberta-large` + SBERT 相似度计算）：~1883 MiB

结论：线程并行模式下（单进程共享 VRAM）**整套评测常驻显存约 1.9GB**；若改回多进程则近似按 worker 数线性增长（例如 5 workers ≈ 9.5GB，会超过 8GB）。

## 运行记录（待填）
### Exp1（HiGMem flat baseline）
- 已完成（2026-02-19 18:17 开始，约 19:21 结束）：
  - 命令：`H-Mem\.venv\Scripts\python.exe -u run_fphm_evaluation.py --dataset data/locomo10.json --model gpt-4o-mini --backend openai --ablation-no-profile --ablation-no-event --k_turn 100 --num-workers 10 --parallel-backend thread`
  - 输出目录：`H-Mem/fphm_runs/gpt-4o-mini_no_event_no_profile_kturn100_sync_20260219_181755`
  - 汇总结果（F1）：`H-Mem/fphm_runs/gpt-4o-mini_no_event_no_profile_kturn100_sync_20260219_181755/aggregated_results.json`
  - 运行日志：`H-Mem/run_logs/higmem_exp1_flat_noevent_kturn100_w10_20260219_181749.*.txt`
  - 观测：初期出现少量 `502 Service temporarily unavailable`，由 retry/backoff 自动重试后正常完成
  - 关键结果（locomo10；n=1982，忽略 evidence 为空的 4 题）：
    - Overall F1=0.4673（Cat1=0.2921, Cat2=0.3767, Cat3=0.1158, Cat4=0.4607, Cat5=0.7312）
    - Recall@K（K=最终选中 turns 数）：Recall=0.7030，Precision=0.0656，avg K=22.7053
    - Token/Time（QA 阶段）：avg token/question=8732.16；avg time/question=7.02s；avg answer prompt tokens=1832.08

### Exp2（A-Mem Recall@fixedK）
- 脚本：`AgenticMemory/test_advanced.py`
- 已中止一次（为提高并发，改为 10 workers）：
  - 旧命令（8 workers）：`AgenticMemory\.venv\Scripts\python.exe -u test_advanced.py --dataset data/locomo10.json --model gpt-4o-mini --backend openai --ratio 1.0 --retrieve_k 50 --num_workers 8 --parallel_backend thread --output results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w8_20260219_172327.json`

- 已完成（2026-02-19 17:34:14 开始；sample4 于 17:55 左右结束；随后汇总输出）：
  - 命令：`AgenticMemory\.venv\Scripts\python.exe -u test_advanced.py --dataset data/locomo10.json --model gpt-4o-mini --backend openai --ratio 1.0 --retrieve_k 50 --num_workers 10 --parallel_backend thread --output results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408.json`
  - API：`OPENAI_API_KEY` 环境变量注入；`api_base` 读取自 `OPENAI_API_BASE`（OpenAI-compatible，结尾 `/v1`）
  - 输出 JSON：`AgenticMemory/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408.json`
  - QA traces（用于 Recall@fixedK）：`AgenticMemory/qa_traces/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408/qa_trace_sample_*.jsonl`
  - stdout：`AgenticMemory/run_logs/amem_full_w10_20260219_173408.out.txt`
  - stderr：`AgenticMemory/run_logs/amem_full_w10_20260219_173408.err.txt`
- Recall@fixedK 分析脚本：
  - 命令：`AgenticMemory\.venv\Scripts\python.exe AgenticMemory/analyze_recall_fixed_k.py --trace_dir AgenticMemory/qa_traces/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408`
  - 输出：`AgenticMemory/qa_traces/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408/recall_fixed_k_summary.csv`
- 关键结果（Macro avg；按“回答 LLM 实际看到的 expanded turns”计 K；忽略 evidence 为空的 4 题，n=1982）：
  - A-Mem turns=8:  Precision=0.0591  Recall=0.3852
  - A-Mem turns=16: Precision=0.0377  Recall=0.4780
  - A-Mem turns=32: Precision=0.0235  Recall=0.5804
  - A-Mem turns≈100(full, avg=99.84): Precision=0.0101  Recall=0.7502
  - HiGMem turns≈12（论文口径 config=`gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync`；natural avg=12.1877）：Precision=0.1336  Recall=0.7469（来源：`H-Mem/analysis_results/recall_and_cost_summary_BEST_49.csv` 的 Overall 行）

### Exp4（Token Cost）
- 脚本/口径：
  - HiGMem：来自 `H-Mem/analyze_recall.py` 的汇总（tiktoken 统计日志中的 prompt/raw_response；只统计 user prompt 文本，不含 system message）
    - 汇总表：`H-Mem/analysis_results/recall_and_cost_summary_BEST_49.csv`（config=`gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync`）
  - A-Mem：来自评测输出 JSON 中的 API token 统计（prompt/completion/total）；并用 tiktoken 估计“关键词生成 prompt tokens”以近似拆分 retrieval vs answer
    - 评测输出：`AgenticMemory/results-amem_gpt-4o-mini_openai_locomo10_ratio1_k50_thread_w10_20260219_173408.json`
  - 自动化汇总脚本：`H-Mem/exp4_token_cost.py`
- 输出：
  - `H-Mem/analysis_results/exp4_token_cost_report.json`
- 关键结果（locomo10 全量；生成阶段=最终 answer prompt 的 input tokens）：
  - A-Mem：avg QA prompt tokens ≈ 12840；其中 answer prompt ≈ 12768（retrieval keywords prompt ≈ 72）
  - HiGMem（event_meta_no_profile；kevent10_noqrw）：avg QA prompt tokens ≈ 15111；其中最终 answer prompt ≈ 1177（retrieval prompt ≈ 13934）
  - 结论：HiGMem 将“下游回答 LLM 的输入 tokens”从 ~12.8k/题 降到 ~1.18k/题（约 10.8× 减少），以 retrieval 阶段额外 token 为代价
- 口径备注：
  - A-Mem tokens 来自其 results JSON 的 `performance_stats.qa`（覆盖 1986 题）
  - HiGMem tokens 来自 `analyze_recall.py` 对日志的 tiktoken 汇总（本表按 recall 统计口径，忽略 evidence 为空的 4 题 → 1982 题）
- Cost 模拟（QA 阶段；gpt-4o-mini: in $0.075/M, out $0.3/M；gpt-5: in $0.625/M, out $5.0/M；详见 json）：
  - gpt-4o-mini：A-Mem ≈ $1.93；HiGMem ≈ $2.40（估计）
  - gpt-5：A-Mem ≈ $16.23；HiGMem ≈ $21.35（估计）
  - Hybrid（retrieval 用 mini，answer 用 gpt-5，仅算 input）：A-Mem ≈ $15.86；HiGMem ≈ $3.53（估计）

### Exp5（Scaling）
- 脚本/口径：
  - `H-Mem/exp5_scaling_analysis.py`
  - 口径说明：
    - Storage(MB)：只估算“记忆数据本身”的 UTF-8 文本字节数 + embedding 矩阵字节数（float32），不包含 Python 对象开销/模型权重/检索器 corpus 等派生缓存
    - Retrieval Time(ms)：只测“向量检索计算”（cosine top-k 的矩阵乘 + argpartition），不包含 query embedding 编码，也不包含任何 LLM 调用/网络时间
- 输出：
  - LoCoMo 前缀点（10/50/100/200 turns）：`H-Mem/analysis_results/exp5_scaling_locomo10.csv`
  - Synthetic 大规模（embeddings-only）：`H-Mem/analysis_results/exp5_scaling_simulated.csv`
- 关键结果（locomo10，10 samples 平均；vector-only）：
  - prefix=10：HiGMem events≈2.3，storage≈0.034MB，retrieval≈0.0058ms；A-Mem storage≈0.020MB，retrieval≈0.0027ms
  - prefix=50：HiGMem events≈5.6，storage≈0.136MB，retrieval≈0.0477ms；A-Mem storage≈0.100MB，retrieval≈0.0393ms
  - prefix=100：HiGMem events≈8.9，storage≈0.258MB，retrieval≈0.0423ms；A-Mem storage≈0.200MB，retrieval≈0.0382ms
  - prefix=200：HiGMem events≈15.8，storage≈0.501MB，retrieval≈0.0427ms；A-Mem storage≈0.400MB，retrieval≈0.0395ms
  - Synthetic（embeddings-only；1K/10K/100K/1M）：
    - 1K turns：A-Mem storage≈1.536MB，retrieval≈0.0457ms；HiGMem(≈85 events) storage≈1.667MB，retrieval≈0.0806ms
    - 10K turns：A-Mem storage≈15.36MB，retrieval≈0.1519ms；HiGMem(≈850 events) storage≈16.67MB，retrieval≈0.1997ms
    - 100K turns：A-Mem storage≈153.6MB，retrieval≈4.51ms；HiGMem(≈8.5K events) storage≈166.7MB，retrieval≈4.79ms
    - 1M turns：A-Mem storage≈1536MB，retrieval≈47.70ms；HiGMem(≈85K events) storage≈1666.56MB，retrieval≈53.04ms

### Exp3（DialSim）
- 状态：**已完成** DialSim v1.1 的 streaming 10K-interaction simulation（turn-level 构筑 + scene-end QA；多选题输出选项原文）。
  - Manifest（固定 seed=0，HiGMem 与 A‑Mem 共用，保证公平）：`H-Mem/dialsim_manifests/dialsim_v1.1_stream_t7000_q3000_seed0.json`
  - 实际有效 turns（从输出 `memory_state.turns_seen` 反推）：friends=2328, bigbang=2318, theoffice=2321 → total=6967
  - 报告（F1 + Recall/turns + token/time）：`H-Mem/analysis_results/exp3_dialsim_report.md`
  - 汇总 JSON（可复现）：
    - HiGMem：`H-Mem/analysis_results/exp3_dialsim_higmem_kevent10_noqrw_summary.json`
    - A‑Mem：`H-Mem/analysis_results/exp3_dialsim_amem_stream_k50_summary.json`
- 运行产物（3000 questions）：
  - HiGMem run dir：`H-Mem/dialsim_runs/gpt-4o-mini_openai_dialsim_stream_kevent10_noqrw_20260222_001838/`
  - A‑Mem run dir：`AgenticMemory/dialsim_runs/gpt-4o-mini_openai_amem_dialsim_stream_20260222_001841/`
- 核心数值（overall；详见报告/汇总 JSON）：
  - HiGMem：F1=0.4152；avgK=3.850；Precision@K=0.1248；Recall@K=0.0864；avg QA prompt tokens=7847.93；avg QA time=4.74s
  - A‑Mem：F1=0.4879；avgK=95.815；Precision@K=0.0426；Recall@K=0.3277；avg QA prompt tokens=12236.60；avg QA time=3.52s
- Hybrid 成本模拟（retrieval=mini, answer=gpt-5，仅算 input；对齐 Exp4 写法）：
  - 报告：`H-Mem/analysis_results/exp3_dialsim_hybrid_cost_report.json`
  - A‑Mem ≈ $22.80 vs HiGMem ≈ $2.54（总 3000 题；约 9× 更省）
- 注：本轮 **不计算 SBERT Similarity**（会触发 HuggingFace 下载；当前 rebuttal 只需要 F1 + 检索 proxy + token/time）。
- 数据格式确认（已通过解包/读取 pickle 验证）：
  - DialSim zip 使用 **Deflate64** 压缩，Python 标准库 `zipfile` 不能直接读；已在两套工程各自 `third_party/zipfile_deflate64/` vendor 了 `zipfile-deflate64` 并在 loader 中自动加载。
  - `*_dialsim.pickle` 结构：`show -> episode -> scene_id -> {date, script, easy_q, hard_q}`。其中 `script` 是一个场景片段（多行 “Speaker: utterance”），`easy_q` 是多选 QA。
  - `v1.0` vs `v1.1`：`easy_q` 计数一致（因此 Table2 若只评 easy，版本影响很小）；`hard_q` 在 v1.1 额外包含更多 temporal 组合类问题（past_fu / fu_fu 等）。
- 公平性：HiGMem 与 A-Mem 的 DialSim QA 使用**完全一致**的多选 prompt（要求输出“选项原文”，不解释），避免“prompt 不一致”导致的偏差。
- 运行脚本：
  - HiGMem：`H-Mem/run_dialsim_streaming_eval.py`（输出 `H-Mem/dialsim_runs/.../predictions.jsonl`）
  - A‑Mem：`AgenticMemory/run_dialsim_streaming_eval.py`（输出 `AgenticMemory/dialsim_runs/.../predictions.jsonl`）
  - 分析（Table2 指标 + 额外 proxy 检索统计）：`H-Mem/analyze_dialsim_results.py`
- 已跑的 sanity check（2026-02-20）：
  - HiGMem（1 scene/show，1 QA/scene，no-event+no-filter 以降低调试成本）：`H-Mem/dialsim_runs/gpt-4o-mini_openai_dialsim_20260220_002448/`
  - A-Mem（同规模）：`AgenticMemory/dialsim_runs/gpt-4o-mini_openai_amem_dialsim_20260220_002833/` 与 `..._20260220_004018/`
  - 注意：该 sanity check 仅用于验证“数据加载 + prompt + 输出格式 + 统计脚本”全链路可用，不代表最终 DialSim 指标。

## 2026-02-21 k_event=10 修复 + Query Rewriting 开关测试（进行中）

### 修复点（必须重跑 Exp3/4/5）
- 论文口径 `k_event=10`，已将默认值修正为 10：
  - `H-Mem/run_fphm_evaluation.py`
  - `H-Mem/run_dialsim_streaming_eval.py`
  - `H-Mem/run_dialsim_evaluation.py`
- 为避免缓存/目录混淆，HiGMem 的 `config_name` 现在显式包含：
  - `kevent{K}`（影响 memory construction：event affiliation）
  - `qrw/noqrw`（是否启用 LLM query rewriting；只影响 retrieval 阶段成本与效果）

### DialSim 行为一致性（locomo10 pipeline）
- `H-Mem/run_dialsim_streaming_eval.py`：已将 **locomo10-consistent 的 original retrieval** 设为默认；
  之前的 fast retrieval 仅在显式 `--use-fast-retrieval` 时启用（不用于论文补实验跑分）。

### Running: locomo10 sample0 query rewriting toggle
- 目的：用 locomo10 sample0 对比 `qrw` vs `noqrw` 的 F1 / token / time / recall / avg turns，决定 DialSim 1w 与后续重跑是否关闭 rewriting。
- 当前已启动（qrw=ON）：
  - 命令：`H-Mem\.venv\Scripts\python.exe run_fphm_evaluation.py --model gpt-4o-mini --backend openai --ablation-no-profile --ablation-event-metadata-only --sample_index 0`
  - stdout/stderr：`H-Mem/run_logs/locomo10_sample0_kevent10_qrw_20260221_152623.(out|err)`
  - jsonl log：`H-Mem/fphm_logs/run_gpt-4o-mini_event_meta_no_profile_kevent10_qrw_sync_*.jsonl`
  - checkpoint：`H-Mem/checkpoints/checkpoint_gpt-4o-mini_event_meta_no_profile_kevent10_qrw_sync_final.pkl`
- 计划：qrw 完成后会复制 checkpoint 到 `noqrw` 配置名下以复用 memory build，再跑 `--disable_query_rewriting_llm` 的 noqrw QA。

### 2026-02-21 API Base 误用修复（重要）
- 现象：`H-Mem/.env` 中 `OPENAI_API_BASE` 误设为 `https://sg.uiuiapi.com/v1`，导致后续 locomo10 全量 run 实际调用了 uiui（现已无额度）。
- 影响：`H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync_20260221_212252/` 属于 **错误 API** 下的部分产物，不能作为有效实验结果引用。
- 处理：已停止相关进程（PID 28564/40780），并将 `H-Mem/.env` 修正为实验统一使用的 `OPENAI_API_BASE`（充足额度）。
- 产物隔离：已将本次错误 API 产生的 checkpoints/results/logs/partial runs 统一移动到：
  - `H-Mem/_invalid_uiui_api_20260221_233129/`
- 重跑：已于 2026-02-21 23:33 启动 **正确 API** 下的 HiGMem locomo10 全量（kevent10 + noqrw + thread + w10）：
  - 运行目录：`H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync_20260221_233315/`
  - 监控：`H-Mem/run_logs/higmem_locomo10_full_kevent10_noqrw_w10_20260221_233305.(out|err)`
  - 注：该目录仅包含每个 sample 的中间产物；最终有效的全量汇总在：`H-Mem/fphm_runs/gpt-4o-mini_event_meta_no_profile_kevent10_noqrw_sync_20260222_003900/aggregated_results.json`
