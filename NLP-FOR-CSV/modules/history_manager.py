# modules/history_manager.py

import pandas as pd
import os
from datetime import datetime

class HistoryManager:
    def __init__(self):
        self.history_file = "analysis_history.csv"
        self.columns = [
            "timestamp",
            "task",
            "prompt",
            "success",
            "confidence",
            "insights"
        ]
        self.load_history()

    # -------------------------------------------------
    def load_history(self):
        if os.path.exists(self.history_file):
            self.history_df = pd.read_csv(self.history_file)
        else:
            self.history_df = pd.DataFrame(columns=self.columns)

    # -------------------------------------------------
    def add_entry(self, task, prompt, results):
        success = bool(results.get("success", False))
        confidence = 1.0 if success else 0.0

        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "prompt": prompt,
            "success": success,
            "confidence": confidence,
            "insights": "; ".join(results.get("insights", []))
        }

        self.history_df = pd.concat(
            [self.history_df, pd.DataFrame([new_entry])],
            ignore_index=True
        )

        self.history_df.to_csv(self.history_file, index=False)

    # -------------------------------------------------
    def add(self, task, params, results):
        """
        Unified logger for all modules
        """
        prompt = (
            params.get("filter_text")
            or params.get("viz_text")
            or params.get("cluster_text")
            or params.get("stats_text")
            or params.get("corr_text")
            or ""
        )

        self.add_entry(
            task=task,
            prompt=prompt,
            results=results
        )

    # -------------------------------------------------
    def get_history(self):
        return self.history_df
