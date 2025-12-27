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
            "hitl_approved",
            "hitl_feedback",
            "rows_retained"
        ]
        self.load_history()

    # -------------------------------------------------
    def load_history(self):
        if os.path.exists(self.history_file):
            self.history_df = pd.read_csv(self.history_file)
        else:
            self.history_df = pd.DataFrame(columns=self.columns)

    # -------------------------------------------------
    def add_hitl_entry(self, task, prompt, approved, feedback, rows_retained):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task,
            "prompt": prompt,
            "hitl_approved": approved,
            "hitl_feedback": feedback,
            "rows_retained": rows_retained
        }

        self.history_df = pd.concat(
            [self.history_df, pd.DataFrame([entry])],
            ignore_index=True
        )

        self.history_df.to_csv(self.history_file, index=False)

    # -------------------------------------------------
    def get_history(self):
        self.load_history()
        return self.history_df
