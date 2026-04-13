# simple_rag_filter.py
import argparse
import json
import os
from tqdm import tqdm
import pickle

from memory_layer import LLMController
from load_dataset import load_locomo_dataset
from utils import calculate_metrics, aggregate_metrics
from fphm_core import FPHMSystem
import prompts

SAMPLE_INDEX = 0
K_TURN_RAG = 100
DATASET_PATH = "data/locomo10.json"
ABLATION_NO_EVENT = True
ABLATION_NO_LINK = True
DEMO_CHECKPOINT_DIR = "demo_checkpoints"
DEMO_LOG_DIR = "demo_logs"

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
        response_str = llm_controller.llm.get_completion(prompt,
                                                         response_format={"type": "json_schema", "json_schema": schema})
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


def build_category_prompt(category: int, context: str, question: str, qa=None) -> str:
    import random
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
                        Based on the context: {context}, write an answher in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
                        Question: {question} Short answer:
                        """
    else:
        prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
                            Question: {question} Short answer:
                            """
    return prompt.strip()


def main():
    parser = argparse.ArgumentParser(
        description="一个简单的脚本，使用RAG和LLM过滤来回答关于对话的问题。"
                    "它首先从对话历史中检索100条最相关的文本片段，然后让LLM根据这些片段生成最终答案。"
    )

    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="用于最终答案生成的LLM模型。")
    parser.add_argument("--backend", type=str, default="openai", help="LLM后端服务 (例如 'openai')。")
    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="LLM后端的API key。")
    parser.add_argument("--api_base", type=str, default=os.getenv("OPENAI_API_BASE"), help="LLM后端的API base URL。")
    args = parser.parse_args()

    samples = load_locomo_dataset(DATASET_PATH)

    sample = samples[SAMPLE_INDEX]
    print(f"已加载样本 {SAMPLE_INDEX}。")

    llm_controller = LLMController(backend=args.backend, model=args.model, api_key=args.api_key, api_base=args.api_base)

    memory_system = FPHMSystem(
        llm_controller=llm_controller,
        run_name=f"simple_rag_filter_{args.model.replace('/', '_')}",
        ablation_no_event=ABLATION_NO_EVENT,
        ablation_no_link=ABLATION_NO_LINK,
        log_dir=DEMO_LOG_DIR,
    )

    checkpoint_filename = f"checkpoint_sample_{SAMPLE_INDEX}_k_{K_TURN_RAG}.pkl"
    checkpoint_path = os.path.join(DEMO_CHECKPOINT_DIR, checkpoint_filename)

    if os.path.exists(checkpoint_path):
        print(f"发现 Checkpoint，正在从 '{checkpoint_path}' 加载索引...")
        with open(checkpoint_path, "rb") as f:
            loaded_state = pickle.load(f)
            memory_system.__dict__.update(loaded_state)
        print("索引加载完成。")
    else:
        print(f"未发现 Checkpoint，开始预处理对话并建立索引")
        all_turns = []
        for session_id in sorted(sample.conversation.sessions.keys()):
            session = sample.conversation.sessions[session_id]

            sorted_session_turns = sorted(session.turns, key=lambda t: int(t.dia_id.split(':')[1]))
            for turn in sorted_session_turns:
                all_turns.append((turn, session.date_time))

        for turn, date_time in tqdm(all_turns, desc="为对话建立索引"):
            memory_system.add_turn(
                turn_id=turn.dia_id,
                turn_content=turn.text,
                speaker=turn.speaker,
                timestamp=date_time
            )
        print("对话索引构建完成。")

        print(f"正在保存 Checkpoint 到 '{checkpoint_path}'...")
        os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)

        state_to_save = memory_system.__dict__.copy()
        if 'llm' in state_to_save: del state_to_save['llm']
        if 'logger' in state_to_save: del state_to_save['logger']
        if 'executor' in state_to_save: del state_to_save['executor']

        with open(checkpoint_path, "wb") as f:
            pickle.dump(state_to_save, f)
        print("Checkpoint 保存成功。下次运行将直接加载此文件。")

    print(f"开始对样本 {SAMPLE_INDEX} 的问题进行评估")
    all_metrics = []
    all_categories = []

    for qa in tqdm(sample.qa, desc="回答问题"):
        query_data = generate_keyword_query(llm_controller, qa.question)
        keyword_query = query_data["keyword_query"]

        context = memory_system.retrieve_for_query(
            original_query=qa.question,
            keyword_query=keyword_query,
            profile_retrieval_keys=[],
            k_profile=0,
            k_event=0,
            k_turn=K_TURN_RAG
        )

        final_prompt = build_category_prompt(
            category=qa.category, context=context, question=qa.question, qa=qa
        )

        answer_json = memory_system._get_llm_json_response(
            final_prompt,
            {"name": "response",
             "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}},
            caller='final_answer_generation', temperature=0.1
        )
        prediction = answer_json.get("answer", "无法生成答案。") if answer_json else "无法生成答案。"

        metrics = calculate_metrics(prediction, qa.final_answer)
        all_metrics.append(metrics)
        all_categories.append(qa.category)

    print("\n--- 评估完成 ---")
    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    print(json.dumps(aggregate_results, indent=2))

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    result_filename = f"results_simple_rag_filter_{args.model.replace('/', '_')}.json"
    result_file_path = os.path.join(results_dir, result_filename)
    with open(result_file_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate_results, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存至: {result_file_path}")


if __name__ == "__main__":
    main()
