import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

def load_sample_data() -> Tuple[pd.DataFrame, Dict]:
    """Sample customer data for testing"""
    np.random.seed(42)
    n = 1000
    data = {
        'customer_id': range(1, n+1),
        'age': np.random.randint(18, 70, n),
        'income': np.random.lognormal(10, 0.5, n).round(0),
        'price': np.random.uniform(10, 500, n).round(2),
        'rating': np.random.uniform(1, 5, n).round(1),
        'sales': np.random.exponential(100, n).round(0),
        'churn': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Food'], n)
    }

    df = pd.DataFrame(data)

    validation = validate_dataset(df)
    validation["source"] = "sample"

    return df, validation

def load_dataset(file) -> Tuple[pd.DataFrame, Dict]:
    """Load CSV/Excel + return validation report"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            raise ValueError("Only CSV/Excel supported")
        
        # Generate validation report
        validation = validate_dataset(df)
        return df, validation
        
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def validate_dataset(df: pd.DataFrame) -> Dict:
    """Comprehensive dataset validation"""
    report = {
        "shape": df.shape,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "missing_count": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / df.size * 100, 2),
        "duplicates": int(df.duplicated().sum()),
        "numeric_cols": list(df.select_dtypes('number').columns),
        "categorical_cols": list(df.select_dtypes('object').columns),
        "quality_score": 0.0,
        "status": ""
    }
    
    # Calculate quality score
    score = 0
    if report["missing_pct"] < 5: score += 25
    if report["duplicates"] == 0: score += 25
    if len(report["numeric_cols"]) > 0: score += 25
    if len(df) > 10: score += 25
    
    report["quality_score"] = score
    report["status"] = "🟢 Good" if score > 75 else "🟡 Fair" if score > 50 else "🔴 Poor"
    
    return report

def suggest_analysis(df: pd.DataFrame) -> List[str]:
    """Generate dataset-specific analysis suggestions"""
    numeric_cols = df.select_dtypes('number').columns.tolist()
    suggestions = []
    
    if len(numeric_cols) >= 1:
        col = numeric_cols[0]
        median_val = df[col].median()
        suggestions.extend([
            "describe the dataset",
            f"filter {col} > {int(median_val)}",
            f"show {col} distribution"
        ])
    
    if len(numeric_cols) >= 2:
        col1, col2 = numeric_cols[:2]
        suggestions.extend([
            f"plot {col1} vs {col2}",
            f"correlation between {col1} and {col2}"
        ])
    
    if len(df) > 50:
        suggestions.append("cluster data k=3")
    
    return suggestions[:6]
