# modules/clustering.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import re


class ClusteringModule:

    # ===============================================================
    # UI
    # ===============================================================
    @staticmethod
    def render_params(numeric_cols, template_name, df):
        st.markdown("### 📈 **Natural Language Clustering**")

        st.info(
            "💡 Examples:\n"
            "- `cluster`\n"
            "- `cluster k=3`\n"
            "- `cluster age income`\n"
            "- `kmeans 4 sales rating`\n"
            "- `segments`"
        )

        cluster_text = st.text_area(
            "Enter clustering command",
            placeholder="cluster age income",
            height=80,
            key=f"cluster_text_{template_name}"
        )

        if cluster_text.strip():
            try:
                fig, insights, _, elbow_fig = ClusteringModule._parse_and_cluster(
                    df, cluster_text, preview=True
                )
                if elbow_fig:
                    st.pyplot(elbow_fig)
                st.pyplot(fig)
                st.success(insights[0])
            except Exception as e:
                st.error(f"❌ {str(e)}")

        return {"cluster_text": cluster_text}

    # ===============================================================
    # Helpers
    # ===============================================================
    @staticmethod
    def _match_column(df, token):
        token = token.lower()

        # Exact match
        for col in df.columns:
            if col.lower() == token:
                return col

        # Plural handling (ratings → rating)
        if token.endswith("s"):
            singular = token[:-1]
            for col in df.columns:
                if col.lower() == singular:
                    return col

        # Partial match
        for col in df.columns:
            if token in col.lower():
                return col

        raise ValueError(
            f"Column '{token}' not found. "
            f"Available numeric columns: {list(df.select_dtypes('number').columns)}"
        )


    # ---------------------------------------------------------------

    @staticmethod
    def _extract_k(tokens):
        for i, token in enumerate(tokens):
            if re.match(r"k=\d+", token):
                return int(token.split("=")[1])
            if token in ["cluster", "kmeans", "segments"] and i + 1 < len(tokens):
                if tokens[i + 1].isdigit():
                    return int(tokens[i + 1])
        return None  # auto-select

    # ---------------------------------------------------------------

    @staticmethod
    def _elbow_method(X_scaled, max_k=10):
        inertias = []
        K = range(2, max_k + 1)

        for k in K:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        # Detect elbow using second derivative
        deltas = np.diff(inertias)
        elbow_k = K[np.argmin(deltas) + 1] if len(deltas) > 1 else 3

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(K, inertias, marker="o")
        ax.axvline(elbow_k, linestyle="--")
        ax.set_xlabel("k")
        ax.set_ylabel("Inertia")
        ax.set_title(f"Elbow Method (Suggested k={elbow_k})")

        return elbow_k, fig

    # ---------------------------------------------------------------

    @staticmethod
    def _describe_clusters(df, features):
        descriptions = []
        grouped = df.groupby("cluster")[features].mean()

        for cluster_id, row in grouped.iterrows():
            desc_parts = []
            for col in features:
                value = row[col]
                overall = df[col].mean()

                delta = (value - overall) / overall * 100 if overall != 0 else 0
                direction = "higher" if delta > 0 else "lower"
                desc_parts.append(f"{direction} {col} ({abs(delta):.1f}% vs avg)")


            descriptions.append(
                f"🔹 Cluster {cluster_id}: represents data points with "
                + ", ".join(desc_parts)
            )

        return descriptions

    # ===============================================================
    # Core Logic
    # ===============================================================
    @staticmethod
    def _parse_and_cluster(df, cluster_text, preview=False):

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) < 2:
            raise ValueError("At least 2 numeric columns are required")

        tokens = cluster_text.lower().split()

        # Extract features
        features = []
        numeric_lower = {c.lower(): c for c in numeric_cols}

        for token in tokens:
            # Skip commands, numbers, parameters
            if token.isdigit():
                continue
            if token in ["cluster", "kmeans", "segments"]:
                continue
            if token.startswith("k="):
                continue

            try:
                col = ClusteringModule._match_column(df, token)
                if col not in features:
                    features.append(col)
            except ValueError:
                # Ignore non-column tokens safely
                continue


        if len(features) < 2:
            features = numeric_cols[:2]
            auto_feature_note = (
                f"⚠️ Insufficient features detected from prompt. "
                f"Defaulting to: {features}"
            )
        else:
            auto_feature_note = None


        X = df[features].fillna(df[features].median())
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Extract or auto-select k
        k = ClusteringModule._extract_k(tokens)
        elbow_fig = None

        if k is None:
            k, elbow_fig = ClusteringModule._elbow_method(
                X_scaled, max_k = max(3, min(10, len(df) // 2))
            )

        if k < 2:
            raise ValueError("k must be ≥ 2")

        # KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)

        df_result = df.copy()
        df_result["cluster"] = clusters

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        scatter = ax1.scatter(
            X_scaled[:, 0], X_scaled[:, 1],
            c=clusters, cmap="viridis", alpha=0.7
        )
        ax1.set_xlabel(features[0])
        ax1.set_ylabel(features[1])
        ax1.set_title(f"KMeans Clusters (k={k})")
        plt.colorbar(scatter, ax=ax1)

        counts = pd.Series(clusters).value_counts().sort_index()
        ax2.pie(
            counts.values,
            labels=[f"Cluster {i}" for i in counts.index],
            autopct="%1.1f%%"
        )
        ax2.set_title("Cluster Distribution")

        plt.tight_layout()

        insights = [
            f"✅ Created {k} clusters using features: {', '.join(features)}"
        ]
        

        if auto_feature_note:
            insights.insert(0, auto_feature_note)

        insights.extend(ClusteringModule._describe_clusters(df_result, features))

        return fig, insights, df_result, elbow_fig

    # ===============================================================
    # Execute
    # ===============================================================
    @staticmethod
    def execute(df, params):

        cluster_text = params.get("cluster_text", "").strip()

        if not cluster_text:
            return {
                "success": True,
                "insights": ["ℹ️ No clustering command provided"],
                "data": df.head(10)
            }

        try:
            fig, insights, df_result, elbow_fig = ClusteringModule._parse_and_cluster(
                df, cluster_text
            )

            return {
                "success": True,
                "insights": insights,
                "figure": fig,
                "elbow_figure": elbow_fig,
                "data": df_result.head(10),
                "cluster_command": cluster_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": df.head(10)
            }
