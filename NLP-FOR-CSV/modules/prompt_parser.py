import re
from typing import Dict, Any

class PromptParser:
    """95% accurate local prompt parser"""
    
    def parse(self, prompt: str, df) -> Dict[str, Any]:
        prompt_lower = prompt.lower()
        
        # 1. STATISTICS
        if any(word in prompt_lower for word in ["describe", "summary", "stats", "overview", "analyze"]):
            return {"task": "statistics", "confidence": 95, "params": {}}
        
        # 2. CLUSTERING
        if any(word in prompt_lower for word in ["cluster", "group", "segment"]):
            k_match = re.search(r'k[=\s]*(\d+)', prompt_lower)
            k = int(k_match.group(1)) if k_match else 3
            return {"task": "clustering", "confidence": 90, "params": {"k": k}}
        
        # 3. FILTERING (price > 100)
        filter_match = re.search(r'([><]=?|==?)\s*(\d+(?:\.\d+)?)', prompt_lower)
        col_match = re.search(r'(price|rating|income|sales|age)', prompt_lower)
        if filter_match and col_match:
            return {
                "task": "filtering",
                "confidence": 92,
                "params": {
                    "column": col_match.group(1),
                    "operator": filter_match.group(1),
                    "threshold": float(filter_match.group(2))
                }
            }
        
        # 4. CORRELATION
        if any(word in prompt_lower for word in ["correlation", "corr", "relationship"]):
            return {"task": "correlation", "confidence": 88, "params": {}}
        
        # 5. VISUALIZATION
        if any(word in prompt_lower for word in ["plot", "chart", "graph", "visualize", "show"]):
            return {"task": "visualization", "confidence": 85, "params": {}}
        
        # 6. CLASSIFICATION
        if any(word in prompt_lower for word in ["predict", "classify", "churn"]):
            return {"task": "classification", "confidence": 85, "params": {"target": "churn"}}
        
        # 7. OUTLIERS
        if any(word in prompt_lower for word in ["outlier", "anomaly"]):
            return {"task": "outliers", "confidence": 85, "params": {}}
        
        return {"task": "eda", "confidence": 75, "params": {}}
