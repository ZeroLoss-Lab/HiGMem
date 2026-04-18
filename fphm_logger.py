# fphm_logger.py
import json
import os
from datetime import datetime

class FPHMLogger:
    def __init__(self, log_dir="fphm_logs", run_name=""):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"run_{run_name}_{timestamp}.jsonl"
        self.log_file = os.path.join(log_dir, log_filename)
        print(f"Logging to {self.log_file}")

    def log(self, step: str, data: dict):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data
        }
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
            print(f"Problematic data: {log_entry}")
