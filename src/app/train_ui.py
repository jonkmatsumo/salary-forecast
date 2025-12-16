import time
from typing import Any, Dict, List, Optional, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.app.api_client import APIError, get_api_client
from src.app.caching import load_data_cached as load_data
from src.app.config_ui import _reset_workflow_state, render_workflow_wizard
from src.app.service_factories import get_analytics_service, get_training_service
from src.services.workflow_service import get_workflow_providers


def render_data_overview(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Render overview metrics.

    Args:
        df (pd.DataFrame): Data.
        summary (Dict[str, Any]): Summary stats.
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", summary.get("total_samples", 0))
    col2.metric("Unique Locations", summary.get("unique_locations", 0))
    col3.metric("Unique Levels", summary.get("unique_levels", 0))


def render_data_sample(df: pd.DataFrame, summary: Dict[str, Any]) -> None:
    """Render data sample preview.

    Args:
        df (pd.DataFrame): Data.
        summary (Dict[str, Any]): Summary stats.
    """
    st.dataframe(df.head())
    st.caption(f"Shape: {summary.get('shape')}")


def render_salary_distribution(df: pd.DataFrame, target_col: str) -> None:
    """Render salary distribution histogram.

    Args:
        df (pd.DataFrame): Data.
        target_col (str): Target column.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x=target_col, kde=True, ax=ax)
    ax.set_title(f"Distribution of {target_col}")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
    st.pyplot(fig)
    st.write("Statistics:")
    st.dataframe(df[target_col].describe().T)


def render_categorical_breakdown(df: pd.DataFrame) -> None:
    """Render categorical breakdown charts.

    Args:
        df (pd.DataFrame): Data.
    """
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Level Counts**")
        if "Level" in df.columns:
            level_counts = df["Level"].value_counts()
            st.bar_chart(level_counts)
        else:
            st.info("Level column not found in data.")
    with c2:
        st.markdown("**Top 20 Locations**")
        if "Location" in df.columns:
            loc_counts = df["Location"].value_counts().head(20)
            st.bar_chart(loc_counts)
        else:
            st.info("Location column not found in data.")


def render_correlations(df: pd.DataFrame, salary_cols: list) -> None:
    """Render correlation heatmap.

    Args:
        df (pd.DataFrame): Data.
        salary_cols (list): Salary columns.
    """
    num_cols = ["YearsOfExperience", "YearsAtCompany"] + salary_cols
    avail_num_cols = [c for c in num_cols if c in df.columns]

    if len(avail_num_cols) > 1:
        corr = df[avail_num_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
    else:
        st.warning("Need at least 2 numerical columns for correlation analysis.")


def render_training_ui() -> None:
    """Render the model training interface. Returns: None."""
    st.header("Model Training")

    df = None
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        dataset_name = st.session_state.get("training_dataset_name", "Unknown")
        st.success(f"Using loaded data: **{dataset_name}** ({len(df)} rows).")
        if st.button("Use Different File"):
            del st.session_state["training_data"]
            if "training_dataset_name" in st.session_state:
                del st.session_state["training_dataset_name"]
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state["training_data"] = df
                st.session_state["training_dataset_name"] = uploaded_file.name

                st.success(f"Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    if df is None:
        st.info("Please upload a CSV file to begin.")
        return

    with st.expander("Data Analysis", expanded=False):
        api_client = get_api_client()
        use_api = api_client is not None

        if use_api:
            assert api_client is not None
            try:
                summary_response = api_client.get_data_summary(df)
                summary = {
                    "total_samples": summary_response.total_samples,
                    "shape": summary_response.shape,
                }
                for col, count in summary_response.unique_counts.items():
                    summary[f"unique_{col.lower().replace(' ', '_')}"] = count
            except APIError as e:
                st.error(f"Failed to get data summary: {e.message}")
                summary = {}
        else:
            analytics_service = get_analytics_service()
            summary = analytics_service.get_data_summary(df)

        viz_options = [
            "Overview Metrics",
            "Data Sample",
            "Salary Distribution",
            "Categorical Breakdown",
            "Correlations",
        ]

        selected_viz = st.selectbox("Select Visualization", viz_options, key="data_analysis_viz")

        st.markdown("---")

        if selected_viz == "Overview Metrics":
            render_data_overview(df, summary)
        elif selected_viz == "Data Sample":
            render_data_sample(df, summary)
        elif selected_viz == "Salary Distribution":
            salary_cols = [
                c for c in ["BaseSalary", "TotalComp", "Stock", "Bonus"] if c in df.columns
            ]
            if salary_cols:
                target_col = st.selectbox(
                    "Select Component", salary_cols, key="salary_dist_component"
                )
                render_salary_distribution(df, target_col)
            else:
                st.warning("No salary columns found in data.")
        elif selected_viz == "Categorical Breakdown":
            render_categorical_breakdown(df)
        elif selected_viz == "Correlations":
            salary_cols = [
                c for c in ["BaseSalary", "TotalComp", "Stock", "Bonus"] if c in df.columns
            ]
            render_correlations(df, salary_cols)

    st.markdown("---")
    wizard_completed = st.session_state.get("workflow_phase") == "complete"
    config = st.session_state.get("config_override")

    # Validate config exists and is not None/empty
    config_valid = config is not None and config != {} and isinstance(config, dict)

    if not wizard_completed or not config_valid:
        with st.expander("AI-Powered Configuration Wizard", expanded=True):
            if not wizard_completed:
                st.write(
                    "**Required:** Complete the configuration wizard before you can start training."
                )
            elif not config_valid:
                st.write(
                    "**Required:** Configuration is missing or invalid. Please regenerate configuration."
                )
            st.info("Generate optimal configuration using an intelligent multi-step workflow.")

            available_providers = get_workflow_providers()
            if not available_providers:
                available_providers = ["openai", "gemini"]

            provider = st.selectbox(
                "LLM Provider", available_providers, index=0, key="wizard_provider_training"
            )

            result = render_workflow_wizard(df, provider)

            if result:
                st.session_state["config_override"] = result
                st.success(
                    "✅ Configuration generated and applied! You can now proceed with training."
                )
                st.rerun()
    else:
        st.success(
            "✅ Configuration wizard completed. You can now configure training options below."
        )
        if st.button("Re-run Configuration Wizard"):
            _reset_workflow_state()
            st.session_state["config_override"] = None
            st.rerun()

    if not wizard_completed or not config_valid:
        if not wizard_completed:
            st.info(
                "⏳ Please complete the AI-Powered Configuration Wizard above to enable training options."
            )
        else:
            st.error(
                "❌ Configuration is missing or invalid. Please complete the Configuration Wizard to generate a valid configuration."
            )
        return

    st.markdown("---")
    st.subheader("Training Configuration")

    do_tune = st.checkbox("Run Hyperparameter Tuning", value=False)
    num_trials = 20
    if do_tune:
        num_trials = st.number_input("Number of Trials", 5, 100, 20)

    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)
    display_charts = st.checkbox("Show Training Performance Chart", value=True)
    additional_tag = st.text_input("Additional Tag (Optional)", placeholder="e.g. experimental-v1")

    api_client = get_api_client()
    use_api = api_client is not None

    if "training_job_id" not in st.session_state:
        st.session_state["training_job_id"] = None
    if "training_dataset_id" not in st.session_state:
        st.session_state["training_dataset_id"] = None

    job_id = st.session_state["training_job_id"]
    dataset_id = st.session_state["training_dataset_id"]

    if job_id is None:
        if st.button("Start Training (Async)", type="primary"):
            # Assert config is valid (checked above) for mypy
            assert isinstance(config, dict)
            dataset_name = st.session_state.get("training_dataset_name", "Unknown Data")

            try:
                if use_api:
                    assert api_client is not None
                    if dataset_id is None:
                        csv_content = df.to_csv(index=False).encode("utf-8")
                        upload_response = api_client.upload_training_data(
                            csv_content, f"{dataset_name}.csv", dataset_name
                        )
                        dataset_id = upload_response.dataset_id
                        st.session_state["training_dataset_id"] = dataset_id

                    job_response = api_client.start_training(
                        dataset_id,
                        config,
                        remove_outliers=remove_outliers,
                        do_tune=do_tune,
                        n_trials=num_trials if do_tune else None,
                        additional_tag=additional_tag if additional_tag.strip() else None,
                        dataset_name=dataset_name,
                    )
                    job_id = job_response.job_id
                else:
                    training_service = get_training_service()
                    job_id = training_service.start_training_async(
                        df,
                        config,
                        remove_outliers=remove_outliers,
                        do_tune=do_tune,
                        n_trials=num_trials,
                        additional_tag=additional_tag if additional_tag.strip() else None,
                        dataset_name=dataset_name,
                    )

                st.session_state["training_job_id"] = job_id
                st.rerun()
            except APIError as e:
                st.error(f"❌ Failed to start training: {e.message}")
            except ValueError as e:
                st.error(f"❌ Configuration error: {e}")
                st.info("Please regenerate your configuration using the Configuration Wizard.")
            except Exception as e:
                st.error(f"❌ Failed to start training: {e}")

    else:
        status: Optional[Dict[str, Any]] = None
        if use_api:
            assert api_client is not None
            try:
                status_response = api_client.get_training_job_status(job_id)
                status = {
                    "status": status_response.status,
                    "logs": status_response.logs or [],
                    "error": status_response.error,
                    "run_id": status_response.run_id,
                    "result": None,
                    "history": [],
                }
            except APIError as e:
                st.error(f"Failed to get job status: {e.message}")
                st.session_state["training_job_id"] = None
                st.rerun()
        else:
            training_service = get_training_service()
            status = training_service.get_job_status(job_id)

            if status is None:
                st.error("Job not found. Clearing state.")
                st.session_state["training_job_id"] = None
                st.rerun()

        assert status is not None
        state = status["status"]
        st.info(f"Training Status: **{state}**")

        if state in ["QUEUED", "RUNNING"]:
            with st.spinner(
                "Training in progress... (You can switch tabs, but stay in app to see completion)"
            ):
                time.sleep(2)
                st.rerun()

        with st.expander("Training Logs", expanded=(state != "COMPLETED")):
            logs = status.get("logs", [])
            st.code("\n".join(logs) if logs else "No logs available yet.")

        if state == "COMPLETED":
            st.success("Training Finished Successfully!")

            history: List[Dict[str, Any]] = cast(List[Dict[str, Any]], status.get("history", []))
            results_data = []

            for event in history:
                if event.get("stage") == "cv_end":
                    results_data.append(
                        {
                            "Model": event.get("model_name"),
                            "Best Round": event.get("best_round"),
                            "Score": event.get("best_score"),
                        }
                    )

            if results_data:
                res_df = pd.DataFrame(results_data)
                if display_charts:
                    st.line_chart(res_df.set_index("Model")["Score"])
                st.dataframe(res_df.style.format({"Score": "{:.4f}"}))

            run_id = status.get("run_id", "N/A")

            if not use_api:
                forecaster = status.get("result")
                if forecaster:
                    st.session_state["forecaster"] = forecaster
                    st.session_state["current_run_id"] = run_id

            st.info(f"Model logged to MLflow with Run ID: **{run_id}**")
            st.markdown("[Open MLflow UI](http://localhost:5000) to view details.")

            if st.button("Start New Training"):
                st.session_state["training_job_id"] = None
                st.session_state["training_dataset_id"] = None
                st.rerun()

        elif state == "FAILED":
            error_msg = status.get("error", "Unknown error")
            st.error(f"Training Failed: {error_msg}")
            if st.button("Retry"):
                st.session_state["training_job_id"] = None
                st.session_state["training_dataset_id"] = None
                st.rerun()
