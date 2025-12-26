import pandas as pd
import os
from datetime import datetime

class HistoryManager:
    def __init__(self):
        self.history_file = "analysis_history.csv"
        self.load_history()
    
    def load_history(self):
        try:
            if os.path.exists(self.history_file):
                self.history_df = pd.read_csv(self.history_file)
            else:
                self.history_df = pd.DataFrame(columns=['timestamp', 'prompt', 'task', 'confidence'])
        except:
            self.history_df = pd.DataFrame(columns=['timestamp', 'prompt', 'task', 'confidence'])
    
    def add_entry(self, prompt, task_info, results, confidence):
        new_entry = pd.DataFrame([{
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'task': task_info['task'],
            'confidence': confidence,
            'insights': str(results.get('insights', [])),
            'success': results.get('success', False)
        }])
        self.history_df = pd.concat([self.history_df, new_entry], ignore_index=True)
        self.history_df.to_csv(self.history_file, index=False)
    
    def get_history(self):
        return self.history_df
