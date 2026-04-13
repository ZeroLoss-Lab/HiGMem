# count_turns.py
import sys
from pathlib import Path
from load_dataset import load_locomo_dataset, LoCoMoSample
from typing import List
import io


# --- 辅助函数：用于临时重定向并抑制不必要的打印输出 ---
class SuppressPrint:
    """A context manager to suppress stdout."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def count_total_turns(samples: List[LoCoMoSample]) -> int:
    """
    遍历所有样本，计算总的对话轮次数量。

    Args:
        samples: 从 load_locomo_dataset 加载的样本列表。

    Returns:
        数据集中所有对话轮次的总数。
    """
    total_turns = 0
    # 遍历每一个样本 (sample)
    for sample in samples:
        # 遍历该样本中的每一个会话 (session)
        for session in sample.conversation.sessions.values():
            # 将当前会话的轮次数量累加到总数中
            total_turns += len(session.turns)

    return total_turns


if __name__ == "__main__":
    # 假设数据集位于项目根目录下的 'data' 文件夹中
    dataset_path = Path("data") / "locomo10.json"

    if not dataset_path.exists():
        print(f"错误：数据集文件未在以下路径找到: {dataset_path}")
        print("请确保 'locomo10.json' 文件存在于 'data' 文件夹中。")
    else:
        print(f"正在从 '{dataset_path}' 加载数据集并进行统计...")

        # 使用 SuppressPrint 来抑制 load_locomo_dataset 函数中详细的打印信息
        # 这样我们可以只关注最终的总数结果
        with SuppressPrint():
            loaded_samples = load_locomo_dataset(dataset_path)

        # 计算总轮次
        total_turns_count = count_total_turns(loaded_samples)

        # 打印最终结果
        print("\n" + "=" * 40)
        print(f"数据集统计完成。")
        print(f"locomo10.json 中总的对话轮次 (turns) 数量为: {total_turns_count}")
        print("=" * 40)

