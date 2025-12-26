import streamlit as st
import pandas as pd

from modules.dataset_loader import load_dataset, load_sample_data
from modules.history_manager import HistoryManager
from modules.filters import FilterModule

# ---------------------------------------------------------
# App Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Data Agent",
    page_icon="🤖",
    layout="wide"
)

st.markdown(
    '<h1 style="text-align:center; color:#1f77b4;">🤖 Smart Data Analysis Agent</h1>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Main App Class
# ---------------------------------------------------------
class DataAgentApp:
    def __init__(self):
        self.history = HistoryManager()
        self.modules = {
            "🔍 Filter": FilterModule()
            # Future modules plug here
        }

    # -----------------------------------------------------
    def run(self):
        tab1, tab2, tab3 = st.tabs(
            ["📁 Dataset", "🎯 Analysis", "📋 History"]
        )

        with tab1:
            self.dataset_tab()

        with tab2:
            self.analysis_tab()

        with tab3:
            self.history_tab()

    # -----------------------------------------------------
    def dataset_tab(self):
        st.header("📁 Dataset")

        col1, col2 = st.columns([3, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel",
                type=["csv", "xlsx"]
            )

        with col2:
            use_sample = st.checkbox("🧪 Use Sample Data", value=True)

        if use_sample:
            df = load_sample_data()
            st.session_state.df = df
            st.success("✅ Sample dataset loaded")

        elif uploaded_file:
            df, _ = load_dataset(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ {df.shape[0]:,} rows loaded")

        if "df" in st.session_state:
            df = st.session_state.df

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Numeric", len(df.select_dtypes("number").columns))

            st.subheader("📌 Numeric Columns")
            st.json(df.select_dtypes("number").columns.tolist())

    # -----------------------------------------------------
    def analysis_tab(self):
        if "df" not in st.session_state:
            st.warning("👆 Upload or load a dataset first")
            return

        df = st.session_state.df
        numeric_cols = df.select_dtypes("number").columns.tolist()

        st.header("🎯 Modular Analysis")

        task = st.selectbox(
            "Select Task",
            list(self.modules.keys()) + [
                "📊 Statistics",
                "📈 Clustering",
                "📉 Correlation",
                "📊 Visualization"
            ]
        )

        templates = self.get_templates(task)

        if templates:
            template_idx = st.selectbox(
                "Choose Template",
                range(len(templates)),
                format_func=lambda i: f"📋 {templates[i]['name']}"
            )
            template = templates[template_idx]

            if template.get("default_col"):
                st.info(
                    f"💡 Suggested: `{template['default_col']} "
                    f"{template['default_op']} {template['default_val']}`"
                )
        else:
            template = {}

        # -------------------------------------------------
        # Module UI
        # -------------------------------------------------
        if task == "🔍 Filter":
            params = self.modules[task].render_params(
                numeric_cols=numeric_cols,
                template_name=template.get("name", "custom"),
                df=df
            )
            preview = params.get("filter_text", "")
        else:
            st.warning("⏳ Module under development")
            params = {}
            preview = "Module not implemented yet"

        st.text_area(
            "📋 Final Command",
            value=preview,
            height=70
        )

        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button(
                "🚀 RUN ANALYSIS",
                type="primary",
                use_container_width=True
            ):
                results = self.execute_task(task, df, params)
                self.show_results(results)

        with col2:
            st.info(f"**Active Task:** {task}")

    # -----------------------------------------------------
    def get_templates(self, task):
        if task == "🔍 Filter":
            return [
                {
                    "name": "Young customers",
                    "default_col": "age",
                    "default_op": "<",
                    "default_val": 35
                },
                {
                    "name": "High income",
                    "default_col": "income",
                    "default_op": ">",
                    "default_val": 60000
                },
                {
                    "name": "Top spenders",
                    "default_col": "sales",
                    "default_op": ">",
                    "default_val": 100
                },
                {
                    "name": "Custom filter",
                    "default_col": None
                }
            ]
        return []

    # -----------------------------------------------------
    def execute_task(self, task, df, params):
        if task in self.modules:
            return self.modules[task].execute(df, params)

        return {
            "success": False,
            "error": f"{task} module not available"
        }

    # -----------------------------------------------------
    def show_results(self, results):
        if results.get("success"):
            st.success("✅ Analysis Complete")

            if "insights" in results:
                st.subheader("💡 Insights")
                for insight in results["insights"]:
                    st.info(insight)

            if "data" in results:
                st.subheader("📊 Result Preview")
                st.dataframe(
                    results["data"].head(10),
                    use_container_width=True
                )
        else:
            st.error(results.get("error", "Analysis failed"))

    # -----------------------------------------------------
    def history_tab(self):
        st.header("📋 History")
        st.info("Execution history will appear here")


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    DataAgentApp().run()
