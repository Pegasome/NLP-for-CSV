def compute_confidence(task_info, df):
    """Calculate analysis confidence"""
    base_conf = task_info.get('confidence', 75) / 100
    
    col_match = len(set(task_info.get('params', {}).keys()).intersection(df.columns)) / max(1, len(df.columns))
    
    return 0.7 * base_conf + 0.3 * col_match
