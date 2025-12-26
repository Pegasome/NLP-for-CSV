import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AnalysisEngine:
    def run(self, task, df, params):
        """Execute analysis task with safety checks"""
        try:
            if task == 'statistics':
                return self._statistics(df)
            elif task == 'filtering':
                return self._filtering(df, params)
            elif task == 'clustering':
                return self._clustering(df)
            elif task == 'correlation':
                return self._correlation(df)
            elif task == 'visualization':
                return self._visualization(df)
            else:
                return self._eda(df)
        except Exception as e:
            return {"success": False, "error": f"Analysis failed: {str(e)}"}
    
    def _safe_get_column(self, df, col_name, default_col=None):
        """Safely get column or return first numeric column"""
        if col_name in df.columns:
            return col_name
        numeric_cols = df.select_dtypes('number').columns
        if len(numeric_cols) > 0:
            return numeric_cols[0]
        return df.columns[0] if len(df.columns) > 0 else None
    
    def _statistics(self, df):
        fig, ax = plt.subplots(figsize=(12, 8))
        desc = df.describe(include='all').round(2)
        desc.plot(kind='bar', ax=ax)
        plt.title("Dataset Statistics")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.close()
        return {"success": True, "data": desc, "figure": fig, 
                "insights": [f"📊 Analyzed {df.shape[0]} rows, {df.shape[1]} columns"]}
    
    def _filtering(self, df, params):
        """FIXED: Safe column access + operator handling"""
        # SAFE COLUMN SELECTION
        col = self._safe_get_column(df, params.get('column', ''))
        if col is None:
            return {"success": False, "error": "No numeric columns found"}
        
        operator = params.get('operator', '>')
        threshold = params.get('threshold', df[col].median())
        
        # SAFE FILTERING
        mask = pd.Series([False] * len(df))
        if operator == '>': mask = df[col] > threshold
        elif operator == '<': mask = df[col] < threshold
        elif operator == '>=': mask = df[col] >= threshold
        elif operator == '<=': mask = df[col] <= threshold
        elif operator == '==': mask = df[col] == threshold
        
        filtered_df = df[mask]
        
        # VISUALIZATION
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        df[col].hist(ax=ax1, bins=30, alpha=0.7, color='blue')
        ax1.axvline(threshold, color='red', lw=3, label=f'{operator} {threshold}')
        ax1.set_title(f'{col} Distribution')
        ax1.legend()
        
        if len(filtered_df) > 0:
            filtered_df[col].hist(ax=ax2, bins=30, alpha=0.7, color='green')
            ax2.set_title(f'Filtered: {len(filtered_df):,} rows')
        else:
            ax2.text(0.5, 0.5, 'No matches found', ha='center', va='center', fontsize=16)
        
        plt.tight_layout()
        plt.close()
        
        return {
            "success": True,
            "data": filtered_df,
            "figure": fig,
            "insights": [f"✅ Filtered {len(filtered_df):,} / {len(df):,} rows ({col} {operator} {threshold})"]
        }
    
    def _clustering(self, df):
        numeric_cols = df.select_dtypes('number').columns
        if len(numeric_cols) < 2:
            return {"success": False, "error": "Need 2+ numeric columns for clustering"}
        
        X = df[numeric_cols[:3]].fillna(df[numeric_cols[:3]].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        result_df = df.copy()
        result_df['cluster'] = clusters
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Set1(clusters / 3)
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=colors, s=50)
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
        ax.set_title('KMeans Clustering (k=3)')
        plt.tight_layout()
        plt.close()
        
        return {
            "success": True,
            "data": result_df.groupby('cluster').size().to_frame(),
            "figure": fig,
            "insights": ["✅ Created 3 customer segments"]
        }
    
    def _correlation(self, df):
        numeric_cols = df.select_dtypes('number').columns
        if len(numeric_cols) < 2:
            return {"success": False, "error": "Need 2+ numeric columns"}
        
        corr_matrix = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45)
        ax.set_yticklabels(numeric_cols)
        plt.colorbar(im)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.close()
        
        return {"success": True, "data": corr_matrix, "figure": fig, 
                "insights": [f"📉 Correlation matrix for {len(numeric_cols)} variables"]}
    
    def _visualization(self, df):
        numeric_cols = df.select_dtypes('number').columns
        if len(numeric_cols) < 2:
            return self._eda(df)
        
        col1, col2 = numeric_cols[:2]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df[col1], df[col2], alpha=0.6)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'{col1} vs {col2} Scatter Plot')
        plt.tight_layout()
        plt.close()
        
        return {"success": True, "figure": fig, "insights": [f"📈 Scatter plot: {col1} vs {col2}"]}
    
    def _eda(self, df):
        numeric_cols = df.select_dtypes('number').columns
        if len(numeric_cols) == 0:
            return {"success": True, "insights": ["No numeric columns for visualization"]}
        
        col = numeric_cols[0]
        fig, ax = plt.subplots(figsize=(10, 6))
        df[col].hist(bins=30, ax=ax, alpha=0.7, edgecolor='black')
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        plt.tight_layout()
        plt.close()
        
        return {"success": True, "figure": fig, "insights": [f"📊 {col} distribution plotted"]}
