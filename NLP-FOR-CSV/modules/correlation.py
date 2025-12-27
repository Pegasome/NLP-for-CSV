# modules/correlation.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Optional seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class CorrelationModule:
    # ===============================================================
    # UI
    # ===============================================================
    @staticmethod
    def render_params(numeric_cols, template_name, df):
        st.markdown("### 📉 **Natural Language Correlation Analysis**")

        st.info(
            "💡 Examples:\n"
            "- `correlation`\n"
            "- `correlation age income sales`\n"
            "- `corr age sales`\n"
            "- `heatmap`\n"
            "- `pairs age income rating`"
        )

        corr_text = st.text_area(
            "Enter correlation command",
            placeholder="correlation age income sales",
            height=80,
            key=f"corr_text_{template_name}"
        )

        if corr_text.strip():
            try:
                insights, fig = CorrelationModule._parse_and_analyze(
                    df, corr_text, preview=True
                )
                st.pyplot(fig)
                st.success(insights[0])
            except Exception as e:
                st.error(f"❌ {str(e)}")

        return {"corr_text": corr_text}

    # ===============================================================
    # Helpers
    # ===============================================================
    @staticmethod
    def _match_column(df, token):
        token = token.lower()

        for col in df.columns:
            if col.lower() == token:
                return col

        if token.endswith("s"):
            singular = token[:-1]
            for col in df.columns:
                if col.lower() == singular:
                    return col

        for col in df.columns:
            if token in col.lower():
                return col

        raise ValueError(
            f"Column '{token}' not found. "
            f"Available numeric columns: {list(df.select_dtypes('number').columns)}"
        )

    @staticmethod
    def _extract_columns(df, tokens):
        numeric_cols = df.select_dtypes("number").columns
        features = []

        for token in tokens:
            if token in ["correlation", "corr", "heatmap", "pairs"]:
                continue
            try:
                col = CorrelationModule._match_column(df, token)
                if col not in features:
                    features.append(col)
            except ValueError:
                continue

        if not features:
            features = numeric_cols[:4].tolist()

        return features

    # ===============================================================
    # Core Logic
    # ===============================================================
    @staticmethod
    def _parse_and_analyze(df, corr_text, preview=False):
        tokens = corr_text.lower().split()
        command = tokens[0]

        features = CorrelationModule._extract_columns(df, tokens)

        if len(features) < 2:
            raise ValueError("Correlation requires at least 2 numeric columns")

        df_num = df[features]
        corr_matrix = df_num.corr()

        # -------- INSIGHTS --------
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        strong_corr = (
            corr_matrix.where(~mask)
            .stack()
            .abs()
            .sort_values(ascending=False)
        )

        insights = [
            f"✅ **Correlation analysis ({len(features)} × {len(features)})**"
        ]

        for (c1, c2), value in strong_corr.head(5).items():
            direction = "positive" if corr_matrix.loc[c1, c2] > 0 else "negative"
            strength = (
                "strong" if value > 0.7 else
                "moderate" if value > 0.4 else
                "weak"
            )
            insights.append(
                f"🔗 **{c1} ↔ {c2}:** {value:.3f} ({direction}, {strength})"
            )

        # -------- VISUALIZATION --------
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(2, 2)

        # Heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        if HAS_SEABORN:
            sns.heatmap(
                corr_matrix, annot=True, cmap="coolwarm",
                center=0, fmt=".2f", ax=ax1
            )
        else:
            im = ax1.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax1.set_xticks(range(len(features)))
            ax1.set_yticks(range(len(features)))
            ax1.set_xticklabels(features, rotation=45, ha="right")
            ax1.set_yticklabels(features)
            fig.colorbar(im, ax=ax1)

        ax1.set_title("Correlation Heatmap")

        # Strongest pair scatter
        ax2 = fig.add_subplot(gs[0, 1])
        c1, c2 = strong_corr.index[0]
        ax2.scatter(df_num[c1], df_num[c2], alpha=0.6)
        ax2.set_xlabel(c1)
        ax2.set_ylabel(c2)
        ax2.set_title(f"Strongest Pair: {c1} vs {c2}")

        # Distribution of correlations
        ax3 = fig.add_subplot(gs[1, 0])
        corr_vals = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
        ax3.hist(corr_vals, bins=15, edgecolor="black")
        ax3.set_title("Distribution of Correlation Coefficients")
        ax3.set_xlabel("Correlation")
        ax3.set_ylabel("Frequency")

        # 3D scatter (ONLY if >=3 features)
        ax4 = fig.add_subplot(gs[1, 1], projection="3d")
        if len(features) >= 3:
            ax4.scatter(
                df_num[features[0]],
                df_num[features[1]],
                df_num[features[2]],
                c=df_num[features[2]],
                cmap="viridis",
                alpha=0.6
            )
            ax4.set_xlabel(features[0])
            ax4.set_ylabel(features[1])
            ax4.set_zlabel(features[2])
            ax4.set_title("3D Feature Relationship")
        else:
            ax4.text(0.5, 0.5, 0.5, "Need 3+ features", ha="center")

        plt.tight_layout()
        return insights, fig

    # ===============================================================
    # Execute
    # ===============================================================
    @staticmethod
    def execute(df, params):
        corr_text = params.get("corr_text", "").strip()

        if not corr_text:
            return {
                "success": True,
                "insights": ["ℹ️ No correlation analysis requested"],
                "data": df.head(10)
            }

        try:
            insights, fig = CorrelationModule._parse_and_analyze(df, corr_text)
            return {
                "success": True,
                "insights": insights,
                "figure": fig,
                "data": df.head(10),
                "corr_command": corr_text
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": df.head(10)
            }
