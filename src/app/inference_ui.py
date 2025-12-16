from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.app.api_client import APIError, get_api_client
from src.app.service_factories import get_analytics_service, get_inference_service
from src.services.inference_service import InvalidInputError, ModelNotFoundError
from src.services.model_registry import ModelRegistry


def render_model_information_api(
    model_details: Any, run_id: str, runs: List[Dict[str, Any]]
) -> None:
    """Display model metadata and feature information from API.

    Args:
        model_details (Any): Model details response.
        run_id (str): MLflow run ID.
        runs (List[Dict[str, Any]]): List of all runs.
    """
    st.markdown("---")
    st.subheader("Model Information")

    with st.expander("Model Metadata", expanded=True):
        metadata = model_details.metadata
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Run ID:** `{run_id[:8]}`")
            if metadata.start_time:
                if hasattr(metadata.start_time, "strftime"):
                    st.markdown(
                        f"**Training Date:** {metadata.start_time.strftime('%Y-%m-%d %H:%M')}"
                    )
                else:
                    st.markdown(f"**Training Date:** {metadata.start_time}")
            st.markdown(f"**Model Type:** {metadata.model_type}")
        with col2:
            cv_score = metadata.cv_mean_score
            if cv_score is not None:
                cv_score = f"{cv_score:.4f}"
            else:
                cv_score = "N/A"
            st.markdown(f"**CV Score:** {cv_score}")
            st.markdown(f"**Dataset:** {metadata.dataset_name}")

            if metadata.additional_tag:
                st.markdown(f"**Tag:** {metadata.additional_tag}")

    with st.expander("Feature Information", expanded=False):
        schema = model_details.model_schema
        if schema:
            if schema.ranked_features:
                st.markdown("**Ranked Features:**")
                for ranked_feat in schema.ranked_features:
                    levels = ranked_feat.levels or []
                    st.markdown(
                        f"- **{ranked_feat.name}**: {len(levels)} levels - {', '.join(levels[:5])}{'...' if len(levels) > 5 else ''}"
                    )

            if schema.proximity_features:
                st.markdown("**Proximity Features:**")
                for prox_feat in schema.proximity_features:
                    st.markdown(f"- **{prox_feat.name}**: Proximity-based encoding")

            if schema.numerical_features:
                st.markdown("**Numerical Features:**")
                st.markdown(
                    f"- {', '.join(schema.numerical_features[:10])}{'...' if len(schema.numerical_features) > 10 else ''}"
                )

            total_features = (
                len(schema.ranked_features or [])
                + len(schema.proximity_features or [])
                + len(schema.numerical_features or [])
            )
            st.markdown(f"**Total Features:** {total_features}")


def render_model_information(
    model: Any, schema: Any, run_id: str, runs: List[Dict[str, Any]]
) -> None:
    """Display model metadata and feature information.

    Args:
        model (Any): Loaded forecaster model.
        schema (Any): Model schema from InferenceService.
        run_id (str): MLflow run ID.
        runs (List[Dict[str, Any]]): List of all runs from registry.
    """
    st.markdown("---")
    st.subheader("Model Information")

    current_run = None
    for r in runs:
        if r["run_id"] == run_id:
            current_run = r
            break

    with st.expander("Model Metadata", expanded=True):
        if current_run:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Run ID:** `{run_id[:8]}`")
                st.markdown(
                    f"**Training Date:** {current_run['start_time'].strftime('%Y-%m-%d %H:%M')}"
                )
                st.markdown(f"**Model Type:** {current_run.get('tags.model_type', 'XGBoost')}")
            with col2:
                cv_score = current_run.get("metrics.cv_mean_score", "N/A")
                if cv_score != "N/A":
                    try:
                        cv_score = f"{float(cv_score):.4f}"
                    except (ValueError, TypeError):
                        pass
                st.markdown(f"**CV Score:** {cv_score}")
                st.markdown(f"**Dataset:** {current_run.get('tags.dataset_name', 'Unknown Data')}")

                add_tag = current_run.get("tags.additional_tag")
                if not add_tag or add_tag == "N/A":
                    add_tag = current_run.get("tags.output_filename")
                if add_tag and add_tag != "N/A":
                    st.markdown(f"**Tag:** {add_tag}")
        else:
            st.info("Metadata not available")

    with st.expander("Feature Information", expanded=False):
        if schema.ranked_features:
            st.markdown("**Ranked Features:**")
            for col_name in schema.ranked_features:
                encoder = model.ranked_encoders[col_name]
                levels = list(encoder.mapping.keys())
                st.markdown(
                    f"- **{col_name}**: {len(levels)} levels - {', '.join(levels[:5])}{'...' if len(levels) > 5 else ''}"
                )

        if schema.proximity_features:
            st.markdown("**Proximity Features:**")
            for col_name in schema.proximity_features:
                st.markdown(f"- **{col_name}**: Proximity-based encoding")

        if schema.numerical_features:
            st.markdown("**Numerical Features:**")
            st.markdown(
                f"- {', '.join(schema.numerical_features[:10])}{'...' if len(schema.numerical_features) > 10 else ''}"
            )

        st.markdown(f"**Total Features:** {len(schema.all_feature_names)}")


def render_inference_ui() -> None:
    """Render the inference interface. Returns: None."""
    st.header("Salary Inference")

    api_client = get_api_client()
    use_api = api_client is not None

    if use_api:
        assert api_client is not None
        try:
            models = api_client.list_models()
            runs = [
                {
                    "run_id": m.run_id,
                    "start_time": m.start_time,
                    "tags.model_type": m.model_type,
                    "tags.dataset_name": m.dataset_name,
                    "tags.additional_tag": m.additional_tag,
                    "tags.output_filename": None,
                    "metrics.cv_mean_score": m.cv_mean_score,
                }
                for m in models
            ]
        except APIError as e:
            st.error(f"Failed to load models from API: {e.message}")
            return
    else:
        registry = ModelRegistry()
        runs = registry.list_models()

    if not runs:
        st.warning("No trained models found in MLflow. Please train a new model.")
        return

    def fmt_score(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)

    def get_run_label(r: Dict[str, Any]) -> str:
        ts = r["start_time"].strftime("%Y-%m-%d %H:%M")
        m_type = r.get("tags.model_type", "XGBoost")
        d_name = r.get("tags.dataset_name", "Unknown Data")
        add_tag = r.get("tags.additional_tag")
        if not add_tag or add_tag == "N/A":
            add_tag = r.get("tags.output_filename")

        cv_score = fmt_score(r.get("metrics.cv_mean_score", "N/A"))
        r_id = r["run_id"][:8]

        base = f"{ts} | {m_type} | {d_name} | CV:{cv_score} | ID:{r_id}"

        if add_tag and add_tag != "N/A":
            base += f" | {add_tag}"

        return base

    run_options = {get_run_label(r): r["run_id"] for r in runs}

    selected_label = st.selectbox("Select Model Version", options=list(run_options.keys()))

    if not selected_label:
        return

    run_id_raw = run_options[selected_label]
    if not isinstance(run_id_raw, str):
        st.error("Invalid run_id type")
        return
    run_id: str = run_id_raw

    if use_api:
        assert api_client is not None
        if (
            "model_details" not in st.session_state
            or st.session_state.get("current_run_id") != run_id
        ):
            with st.spinner(f"Loading model details from API for run {run_id}..."):
                try:
                    model_details = api_client.get_model_details(run_id)
                    st.session_state["model_details"] = model_details
                    st.session_state["current_run_id"] = run_id
                except APIError as e:
                    st.error(f"Failed to load model details: {e.message}")
                    return
        model_details = st.session_state["model_details"]
        render_model_information_api(model_details, run_id, runs)
    else:
        inference_service = get_inference_service()
        if "forecaster" not in st.session_state or st.session_state.get("current_run_id") != run_id:
            with st.spinner(f"Loading model from MLflow run {run_id}..."):
                try:
                    model = inference_service.load_model(run_id)
                    schema = inference_service.get_model_schema(model)
                    st.session_state["forecaster"] = model
                    st.session_state["forecaster_schema"] = schema
                    st.session_state["current_run_id"] = run_id
                except ModelNotFoundError as e:
                    st.error(f"Failed to load model: {e}")
                    return
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    return
        forecaster = st.session_state["forecaster"]
        schema = st.session_state["forecaster_schema"]
        render_model_information(forecaster, schema, run_id, runs)

    with st.expander("Model Analysis", expanded=False):
        if use_api:
            model_details = st.session_state["model_details"]
            targets = model_details.targets

            if not targets:
                st.warning("No targets available in this model.")
            else:
                viz_options = ["Feature Importance"]
                selected_viz = st.selectbox(
                    "Select Visualization", viz_options, key="model_analysis_viz"
                )

                st.markdown("---")

                if selected_viz == "Feature Importance":
                    selected_target = st.selectbox(
                        "Select Target Component", targets, key="model_analysis_target"
                    )

                    if selected_target:
                        quantiles = model_details.quantiles
                        if quantiles:
                            selected_q_val = st.selectbox(
                                "Select Quantile",
                                quantiles,
                                format_func=lambda x: f"P{int(x*100)}",
                                key="model_analysis_quantile",
                            )

                            with st.spinner("Loading feature importance..."):
                                try:
                                    assert api_client is not None
                                    importance_response = api_client.get_feature_importance(
                                        run_id, selected_target, selected_q_val
                                    )
                                    if importance_response.features:
                                        df_imp = pd.DataFrame(
                                            [
                                                {"Feature": f.name, "Gain": f.gain}
                                                for f in importance_response.features
                                            ]
                                        )
                                        if not df_imp.empty:
                                            st.dataframe(df_imp, width="stretch")

                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            sns.barplot(
                                                data=df_imp.head(20),
                                                x="Gain",
                                                y="Feature",
                                                ax=ax,
                                                palette="viridis",
                                                hue="Feature",
                                                legend=False,
                                            )
                                            ax.set_title(
                                                f"Top 20 Features for {selected_target} (P{int(selected_q_val*100)})"
                                            )
                                            st.pyplot(fig)
                                        else:
                                            st.warning(
                                                f"No feature importance scores found for {selected_target} at P{int(selected_q_val*100)}."
                                            )
                                    else:
                                        st.warning(
                                            f"No feature importance scores found for {selected_target} at P{int(selected_q_val*100)}."
                                        )
                                except APIError as e:
                                    st.error(f"Failed to load feature importance: {e.message}")
                        else:
                            st.warning(f"No quantiles available for target {selected_target}.")
        else:
            forecaster = st.session_state["forecaster"]
            schema = st.session_state["forecaster_schema"]
            analytics_service = get_analytics_service()
            targets = schema.targets

            if not targets:
                st.warning("No targets available in this model.")
            else:
                viz_options = ["Feature Importance"]
                selected_viz = st.selectbox(
                    "Select Visualization", viz_options, key="model_analysis_viz"
                )

                st.markdown("---")

                if selected_viz == "Feature Importance":
                    selected_target = st.selectbox(
                        "Select Target Component", targets, key="model_analysis_target"
                    )

                    if selected_target:
                        quantiles = schema.quantiles
                        if quantiles:
                            selected_q_val = st.selectbox(
                                "Select Quantile",
                                quantiles,
                                format_func=lambda x: f"P{int(x*100)}",
                                key="model_analysis_quantile",
                            )

                            df_imp = analytics_service.get_feature_importance(
                                forecaster, selected_target, selected_q_val
                            )
                            if df_imp is not None and not df_imp.empty:
                                st.dataframe(df_imp, width="stretch")

                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(
                                    data=df_imp.head(20),
                                    x="Gain",
                                    y="Feature",
                                    ax=ax,
                                    palette="viridis",
                                    hue="Feature",
                                    legend=False,
                                )
                                ax.set_title(
                                    f"Top 20 Features for {selected_target} (P{int(selected_q_val*100)})"
                                )
                                st.pyplot(fig)
                            else:
                                st.warning(
                                    f"No feature importance scores found for {selected_target} at P{int(selected_q_val*100)}."
                                )
                        else:
                            st.warning(f"No quantiles available for target {selected_target}.")

    st.markdown("---")

    if use_api:
        model_details = st.session_state["model_details"]
        schema = model_details.model_schema
        if not schema:
            st.error("Model schema not available")
            return

        with st.form("inference_form"):
            st.subheader("Input Features")
            c1, c2 = st.columns(2)

            input_data = {}

            with c1:
                for ranked_feat in schema.ranked_features or []:
                    levels = ranked_feat.levels or []
                    val = st.selectbox(ranked_feat.name, levels, key=f"input_{ranked_feat.name}")
                    input_data[ranked_feat.name] = val

                for prox_feat in schema.proximity_features or []:
                    val = st.text_input(prox_feat.name, "New York", key=f"input_{prox_feat.name}")
                    input_data[prox_feat.name] = val

            with c2:
                handled = [f.name for f in (schema.ranked_features or [])] + [
                    f.name for f in (schema.proximity_features or [])
                ]
                remaining = [f for f in (schema.numerical_features or []) if f not in handled]

                for feat in remaining:
                    val = st.number_input(feat, 0, 100, 5, key=f"input_{feat}")
                    input_data[feat] = val

            submitted = st.form_submit_button("Predict")

        if submitted:
            with st.spinner("Predicting..."):
                try:
                    assert api_client is not None
                    response = api_client.predict(run_id, input_data)
                    results = response.data.predictions if response.data else {}
                except APIError as e:
                    st.error(f"Prediction failed: {e.message}")
                    return

            st.subheader("Prediction Results")

            res_data = []
            for target, preds in results.items():
                row = {"Component": target}
                for q_key, val in preds.items():
                    row[q_key] = val
                res_data.append(row)

            res_df = pd.DataFrame(res_data)
            chart_df = res_df.set_index("Component").T

            try:
                sorted_index = sorted(chart_df.index, key=lambda x: int(x.replace("p", "")))
                chart_df = chart_df.reindex(sorted_index)
            except ValueError:
                pass

            st.line_chart(chart_df)

            st.dataframe(
                res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"})
            )
    else:
        inference_service = get_inference_service()
        forecaster = st.session_state["forecaster"]
        schema = st.session_state["forecaster_schema"]
        with st.form("inference_form"):
            st.subheader("Input Features")
            c1, c2 = st.columns(2)

            input_data = {}

            with c1:
                for col_name in schema.ranked_features:
                    encoder = forecaster.ranked_encoders[col_name]
                    levels = list(encoder.mapping.keys())
                    val = st.selectbox(col_name, levels, key=f"input_{col_name}")
                    input_data[col_name] = val

                for col_name in schema.proximity_features:
                    val = st.text_input(col_name, "New York", key=f"input_{col_name}")
                    input_data[col_name] = val

            with c2:
                for feat in schema.numerical_features:
                    val = st.number_input(feat, 0, 100, 5, key=f"input_{feat}")
                    input_data[feat] = val

            submitted = st.form_submit_button("Predict")

        if submitted:
            with st.spinner("Predicting..."):
                try:
                    result = inference_service.predict(forecaster, input_data)
                    results = result.predictions
                    metadata = result.metadata
                except InvalidInputError as e:
                    st.error(f"Invalid input: {e}")
                    return
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    return

            st.subheader("Prediction Results")

            if metadata.get("location_zone"):
                st.markdown(f"**Target Location Zone:** {metadata['location_zone']}")

            res_data = []
            for target, preds in results.items():
                row = {"Component": target}
                for q_key, val in preds.items():
                    row[q_key] = val
                res_data.append(row)

            res_df = pd.DataFrame(res_data)
            chart_df = res_df.set_index("Component").T

            try:
                sorted_index = sorted(chart_df.index, key=lambda x: int(x.replace("p", "")))
                chart_df = chart_df.reindex(sorted_index)
            except ValueError:
                pass

            st.line_chart(chart_df)

            st.dataframe(
                res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"})
            )
