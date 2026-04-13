# fphm_logger.py
import json
import os
import threading
from datetime import datetime

class FPHMLogger:
    def __init__(self, log_dir="fphm_logs", run_name=""):
        """
        初始化日志记录器。
        :param log_dir: 日志文件存放的目录。
        :param run_name: 本次运行的名称，用于区分不同的实验（如完整版 vs 消融版）。
        """
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_{run_name}_{timestamp}.jsonl"
        self.log_file = os.path.join(log_dir, log_filename)
        # Protect concurrent writes from ThreadPoolExecutor to avoid interleaved / malformed JSONL lines.
        self._write_lock = threading.Lock()
        print(f"Logging to {self.log_file}")

    def log(self, step: str, data: dict):
        """
        记录一个步骤的数据。
        :param step: 步骤名称，如 'decide_event_affiliation'。
        :param data: 要记录的数据，必须是可JSON序列化的字典。
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data
        }
        try:
            line = json.dumps(log_entry, ensure_ascii=False, default=str) + "\n"
            with self._write_lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception as e:
            print(f"Error writing to log file: {e}")
            print(f"Problematic data: {log_entry}")
