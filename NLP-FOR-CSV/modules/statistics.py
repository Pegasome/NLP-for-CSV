# modules/statistics.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# Optional seaborn (safe fallback)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class StatisticsModule:
    # ===============================================================
    # UI
    # ===============================================================
    @staticmethod
    def render_params(numeric_cols, template_name, df):
        st.markdown("### 📊 **Natural Language Statistics**")

        st.info(
            "💡 Examples:\n"
            "- `stats`\n"
            "- `stats age income`\n"
            "- `describe sales rating`\n"
            "- `summary`\n"
            "- `correlation age sales`"
        )

        stats_text = st.text_area(
            "Enter statistics command",
            placeholder="stats age income sales",
            height=80,
            key=f"stats_text_{template_name}"
        )

        if stats_text.strip():
            try:
                insights, fig = StatisticsModule._parse_and_analyze(
                    df, stats_text, preview=True
                )
                if fig:
                    st.pyplot(fig)
                st.success(insights[0])
            except Exception as e:
                st.error(f"❌ {str(e)}")

        return {"stats_text": stats_text}

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

        # Plural handling
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
    def _extract_columns(df, tokens):
        numeric_cols = df.select_dtypes("number").columns
        features = []

        for token in tokens:
            if (
                token.isdigit()
                or token in ["stats", "describe", "summary", "correlation"]
                or token.startswith("k=")
            ):
                continue
            try:
                col = StatisticsModule._match_column(df, token)
                if col not in features:
                    features.append(col)
            except ValueError:
                continue

        if not features:
            features = numeric_cols[:3].tolist()

        return features

    # ---------------------------------------------------------------

    @staticmethod
    def _describe_stats(df_result, features):
        descriptions = []

        for col in features:
            stats = df_result[col].describe()
            mean = stats["mean"]
            std = stats["std"]
            cv = std / mean if mean != 0 else 0

            descriptions.extend([
                f"📈 **{col}:**",
                f"   • Mean: {mean:.2f}",
                f"   • Median: {stats['50%']:.2f}",
                f"   • Std Dev: {std:.2f}",
                f"   • Coefficient of Variation: {cv:.2f}",
                f"   • Range: {stats['min']:.2f} – {stats['max']:.2f}",
                f"   • Missing: {df_result[col].isna().sum()} "
                f"({df_result[col].isna().sum()/len(df_result)*100:.1f}%)"
            ])

        return descriptions

    # ===============================================================
    # Core Logic
    # ===============================================================
    @staticmethod
    def _parse_and_analyze(df, stats_text, preview=False):
        tokens = stats_text.lower().split()
        command = tokens[0]

        features = StatisticsModule._extract_columns(df, tokens)
        df_result = df[features].copy()

        # ---------------- SUMMARY STATS ----------------
        if command in ["stats", "describe", "summary"]:
            insights = [f"✅ **Summary statistics generated for {len(features)} columns**"]
            insights.extend(StatisticsModule._describe_stats(df_result, features))

            fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
            if len(features) == 1:
                axes = [axes]

            for i, col in enumerate(features):
                axes[i].hist(df[col].dropna(), bins=30, alpha=0.7)
                axes[i].set_title(col)
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")

            plt.tight_layout()

        # ---------------- CORRELATION ----------------
        elif command == "correlation":
            if len(features) < 2:
                raise ValueError("Correlation requires at least 2 numeric columns")

            corr_matrix = df_result.corr()
            insights = [f"✅ **Correlation analysis ({len(features)} × {len(features)})**"]

            # Strongest correlations
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            strong_corr = (
                corr_matrix.where(~mask)
                .stack()
                .abs()
                .sort_values(ascending=False)
            )

            for (c1, c2), value in strong_corr.head(5).items():
                direction = "positive" if corr_matrix.loc[c1, c2] > 0 else "negative"
                insights.append(
                    f"🔗 **{c1} ↔ {c2}:** {value:.3f} ({direction})"
                )

            fig, ax = plt.subplots(figsize=(8, 6))

            if HAS_SEABORN:
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    center=0,
                    ax=ax
                )
            else:
                im = ax.imshow(corr_matrix, cmap="coolwarm")
                ax.set_xticks(range(len(features)))
                ax.set_yticks(range(len(features)))
                ax.set_xticklabels(features, rotation=45)
                ax.set_yticklabels(features)
                plt.colorbar(im, ax=ax)

            ax.set_title("Correlation Heatmap")

        else:
            raise ValueError(
                "Supported commands: stats, describe, summary, correlation"
            )

        return insights, fig

    # ===============================================================
    # Execute
    # ===============================================================
    @staticmethod
    def execute(df, params):
        stats_text = params.get("stats_text", "").strip()

        if not stats_text:
            return {
                "success": True,
                "insights": ["ℹ️ No statistics command provided"],
                "data": df.head(10)
            }

        try:
            insights, fig = StatisticsModule._parse_and_analyze(df, stats_text)

            return {
                "success": True,
                "insights": insights,
                "figure": fig,
                "data": df.head(10),
                "stats_command": stats_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": df.head(10)
            }
