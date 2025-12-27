import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re


class VisualizeModule:
    @staticmethod
    def render_params(numeric_cols, template_name, df):
        """📊 Natural Language Visualization UI"""

        st.markdown("### 📈 Natural Language Visualization")

        st.info(
            "💡 Examples:\n"
            "- `scatter age income`\n"
            "- `histogram sales`\n"
            "- `boxplot rating by churn`\n"
            "- `line age`"
        )

        viz_text = st.text_area(
            "Enter visualization command",
            placeholder="scatter age income",
            height=80,
            key=f"viz_text_{template_name}"
        )

        if viz_text.strip():
            try:
                fig = VisualizeModule._parse_and_build(df, viz_text)
                st.pyplot(fig)
                st.success("✅ Preview generated")
            except Exception as e:
                st.error(f"❌ {str(e)}")

        return {"viz_text": viz_text}

    # -------------------------------------------------
    @staticmethod
    def _match_column(df, token):
        """Safely match user token to dataframe column"""
        matches = [c for c in df.columns if c.lower() == token.lower()]
        if not matches:
            raise ValueError(f"Column '{token}' not found")
        return matches[0]

    # -------------------------------------------------
    @staticmethod
    def _parse_and_build(df, viz_text):
        """Parse NL visualization command → matplotlib figure"""

        text = viz_text.lower().strip()
        # -----------------------------------------
        # AUTO SCATTER: "age vs income"
        # -----------------------------------------
        if " vs " in text:
            parts = re.split(r"\s+vs\s+", text)
            if len(parts) == 2:
                x_col = VisualizeModule._match_column(df, parts[0])
                y_col = VisualizeModule._match_column(df, parts[1])

                fig, ax = plt.subplots(figsize=(9, 5))
                ax.scatter(df[x_col], df[y_col], alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{x_col} vs {y_col}")
                plt.tight_layout()
                return fig

        # Normalize spaces
        tokens = re.split(r"\s+", text)
        command = tokens[0]

        fig, ax = plt.subplots(figsize=(9, 5))

        # -----------------------------------------
        # SCATTER
        # -----------------------------------------
        if command == "scatter":
            if len(tokens) < 3:
                raise ValueError("Usage: scatter <x> <y>")

            x_col = VisualizeModule._match_column(df, tokens[1])
            y_col = VisualizeModule._match_column(df, tokens[2])

            ax.scatter(df[x_col], df[y_col], alpha=0.6)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{x_col} vs {y_col}")

        # -----------------------------------------
        # HISTOGRAM
        # -----------------------------------------
        elif command in ["hist", "histogram"]:
            if len(tokens) < 2:
                raise ValueError("Usage: histogram <column>")

            col = VisualizeModule._match_column(df, tokens[1])

            ax.hist(df[col].dropna(), bins=30, alpha=0.7)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
            ax.set_title(f"Histogram of {col}")

        # -----------------------------------------
        # BOXPLOT
        # -----------------------------------------
        elif command == "boxplot":
            if "by" in tokens:
                col_idx = tokens.index("boxplot") + 1
                by_idx = tokens.index("by")

                col = VisualizeModule._match_column(df, tokens[col_idx])
                by_col = VisualizeModule._match_column(df, tokens[by_idx + 1])

                df.boxplot(column=col, by=by_col, ax=ax)
                ax.set_title(f"{col} by {by_col}")
                plt.suptitle("")
            else:
                col = VisualizeModule._match_column(df, tokens[1])
                ax.boxplot(df[col].dropna())
                ax.set_title(f"Boxplot of {col}")
                ax.set_ylabel(col)

        # -----------------------------------------
        # LINE
        # -----------------------------------------
        elif command == "line":
            if len(tokens) > 2:
                raise ValueError(
                    "Line chart supports ONE column only. "
                    "Use scatter for relationships."
                )

            if len(tokens) < 2:
                raise ValueError("Usage: line <column>")

            col = VisualizeModule._match_column(df, tokens[1])
            ax.plot(df[col].values)
            ax.set_title(f"Line plot of {col}")
            ax.set_ylabel(col)
            ax.set_xlabel("Index")

        else:
            raise ValueError(
                "Unsupported visualization. "
                "Supported: scatter, histogram, boxplot, line"
            )

        plt.tight_layout()
        return fig

    # -------------------------------------------------
    @staticmethod
    def execute(df, params):
        """Execute visualization command"""

        viz_text = params.get("viz_text", "").strip()

        if not viz_text:
            return {
                "success": True,
                "insights": ["ℹ️ No visualization requested"],
                "data": df.head(10)
            }

        try:
            fig = VisualizeModule._parse_and_build(df, viz_text)

            return {
                "success": True,
                "insights": [f"📊 Visualization executed: `{viz_text}`"],
                "figure": fig,
                "data": df.head(10),
                "viz_used": viz_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": df
            }
