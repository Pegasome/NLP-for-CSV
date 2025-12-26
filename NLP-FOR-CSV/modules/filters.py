import streamlit as st
import pandas as pd
import re


class FilterModule:
    @staticmethod
    def render_params(numeric_cols, template_name, df):
        """🔍 Natural Language Filter UI"""
        st.markdown("### 🤖 **Natural Language Filter**")

        st.info("💡 Examples: `age > 30 AND income >= 50000`, `age < 40 OR salary > 60000`")

        filter_text = st.text_area(
            "**Enter your filter:**",
            placeholder="age > 30 and income >= 50000",
            height=80,
            key=f"filter_text_{template_name}"
        )

        # 🔥 LIVE PREVIEW
        if filter_text.strip():
            try:
                preview_mask = FilterModule._parse_and_preview(df, filter_text)
                st.success(f"✅ Preview: **{preview_mask.sum():,} / {len(df):,} rows** match")
            except Exception as e:
                st.error(f"❌ Invalid filter syntax: {e}")

        # 🔥 ADVANCED MODE
        if st.checkbox("⚙️ Show column stats", key=f"filter_adv_{template_name}"):
            st.subheader("📊 Column Statistics")
            stats = pd.DataFrame({
                "Column": numeric_cols,
                "Min": [df[c].min() for c in numeric_cols],
                "Median": [df[c].median() for c in numeric_cols],
                "Max": [df[c].max() for c in numeric_cols]
            })
            st.dataframe(stats, use_container_width=True)

        return {"filter_text": filter_text}

    # ---------------------------------------------------------
    # 🔥 Core parser
    # ---------------------------------------------------------
    @staticmethod
    def _parse_and_preview(df, filter_text):
        """Parse NL filter into boolean mask"""

        if not filter_text:
            return pd.Series(True, index=df.index)

        # Normalize
        filter_text = filter_text.strip()
        df_cols = {c.lower(): c for c in df.columns}

        tokens = re.split(r"\s+(AND|OR)\s+", filter_text, flags=re.IGNORECASE)

        final_mask = None
        pending_logic = None

        for token in tokens:
            token_upper = token.upper()

            if token_upper in ("AND", "OR"):
                pending_logic = token_upper
                continue

            # ---- Parse condition ----
            condition = token.strip()

            match = re.match(
                r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(>=|<=|==|>|<|=)\s*([0-9.]+)",
                condition
            )

            if not match:
                raise ValueError(f"Invalid condition: '{condition}'")

            col_raw, op, val = match.groups()
            col = df_cols.get(col_raw.lower())

            if col is None:
                raise ValueError(f"Unknown column: '{col_raw}'")

            val = float(val)

            # ---- Apply operator ----
            if op == ">":
                mask = df[col] > val
            elif op == "<":
                mask = df[col] < val
            elif op == ">=":
                mask = df[col] >= val
            elif op == "<=":
                mask = df[col] <= val
            elif op in ("=", "=="):
                mask = df[col] == val
            else:
                raise ValueError(f"Unsupported operator: {op}")

            # ---- Combine ----
            if final_mask is None:
                final_mask = mask
            else:
                if pending_logic == "OR":
                    final_mask = final_mask | mask
                else:
                    final_mask = final_mask & mask

        return final_mask.fillna(False)

    # ---------------------------------------------------------
    # 🔥 Execution
    # ---------------------------------------------------------
    @staticmethod
    def execute(df, params):
        """Execute natural language filter"""

        filter_text = params.get("filter_text", "").strip()

        if not filter_text:
            return {
                "success": True,
                "insights": ["ℹ️ No filter applied"],
                "data": df
            }

        try:
            mask = FilterModule._parse_and_preview(df, filter_text)
            filtered_df = df.loc[mask].copy()

            insights = [
                f"✅ **{len(filtered_df):,} / {len(df):,} rows** retained",
                f"🔍 Filter: `{filter_text}`",
                f"📊 Numeric columns: {len(df.select_dtypes('number').columns)}"
            ]

            return {
                "success": True,
                "insights": insights,
                "data": filtered_df,
                "filter_used": filter_text
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"❌ Filter parse error: {e}",
                "data": df
            }
