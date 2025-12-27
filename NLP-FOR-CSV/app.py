import streamlit as st
import pandas as pd

from modules.dataset_loader import load_dataset, load_sample_data
from modules.history_manager import HistoryManager
from modules.filters import FilterModule
from modules.visualize import VisualizeModule
from modules.clustering import ClusteringModule
from modules.statistics import StatisticsModule
from modules.correlation import CorrelationModule

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
        self.history_manager = HistoryManager()
        self.modules = {
            "🔍 Filter": FilterModule(),
            "📊 Visualization": VisualizeModule(),
            " Clustering" : ClusteringModule(),
            "Statistics" : StatisticsModule(),
            "Correlation" : CorrelationModule()
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
            df, validation = load_sample_data()
            st.session_state.df = df
            st.session_state.validation = validation
            st.success(f"✅ Sample dataset loaded ({validation['status']})")


        elif uploaded_file:
            df, validation = load_dataset(uploaded_file)
            st.session_state.df = df
            st.session_state.validation = validation
            st.success(
                f"✅ {df.shape[0]:,} rows loaded | "
                f"Quality: {validation['status']}"
            )


        if "df" in st.session_state:
            df = st.session_state.df

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric("Numeric", len(df.select_dtypes("number").columns))
            if "validation" in st.session_state:
                with st.expander("📋 Dataset Quality Report"):
                    st.json(st.session_state.validation)
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
            list(self.modules.keys()) 
        )

        # -------------------------------------------------
        # Module UI
        # -------------------------------------------------
        if task in self.modules:
            params = self.modules[task].render_params(
                numeric_cols=numeric_cols,
                template_name="custom",
                df=df
            )

            preview = (
                params.get("filter_text")
                or params.get("viz_text")
                or ""
            )
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
                st.session_state["last_results"] = self.execute_task(task, df, params)

        # ✅ Always render results if present
        if "last_results" in st.session_state:
            self.show_results(st.session_state["last_results"])

        with col2:
            st.info(f"**Active Task:** {task}")

     # -----------------------------------------------------
    def execute_task(self, task, df, params):
        result = self.modules[task].execute(df, params)

        result["task"] = task
        result["params"] = params
        result["requires_hitl"] = task in ["🔍 Filter", "📦 Clustering"]

        return result

    # -----------------------------------------------------
    def show_results(self, results):
        if not results.get("success"):
            st.error(results.get("error", "Analysis failed"))
            return

        st.success("✅ Analysis Complete")

        if "insights" in results:
            st.subheader("💡 Insights")
            for insight in results["insights"]:
                st.info(insight)

        if "data" not in results:
            return

        st.subheader("📊 Result Preview")

        hitl_key = f"hitl_submitted_{results['task']}"

        if hitl_key not in st.session_state:
            st.session_state[hitl_key] = False

        if results.get("requires_hitl"):
            st.info(
                "🧑‍💼 Human-in-the-Loop Review\n\n"
                "Edits are for audit only. Analysis is NOT re-run."
            )

            edited_df = st.data_editor(
                results["data"].head(50),
                use_container_width=True
            )

            human_approved = st.radio(
                "✅ Approve this data?",
                ["Yes", "No"]
            )

            human_feedback = st.text_area(
                "💬 Optional feedback",
                placeholder="E.g., tighten filter, remove false positives…"
            )

            submit = st.button("Submit HITL Feedback")

            if submit and not st.session_state[hitl_key]:
                st.session_state[hitl_key] = True

                prompt = (
                    results["params"].get("filter_text")
                    or results["params"].get("viz_text")
                    or ""
                )

                self.history_manager.add_hitl_entry(
                    task=results["task"],
                    prompt=prompt,
                    approved=(human_approved == "Yes"),
                    feedback=human_feedback,
                    rows_retained=len(edited_df)
                )

                st.success("✅ HITL decision saved to history")

            if st.session_state[hitl_key]:
                st.info("🔒 HITL already submitted for this analysis")

    # -----------------------------------------------------
    def history_tab(self):
        st.header("📋 History")

        history_df = self.history_manager.get_history()

        if history_df.empty:
            st.info("No history yet.")
            return

        history_df = history_df.sort_values(
            "timestamp", ascending=False
        )

        st.dataframe(history_df, use_container_width=True)


# ---------------------------------------------------------
# Run App
# ---------------------------------------------------------
if __name__ == "__main__":
    DataAgentApp().run()