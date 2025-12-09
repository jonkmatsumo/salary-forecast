import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.services.model_registry import ModelRegistry
from src.services.analytics_service import AnalyticsService


def render_model_information(forecaster: Any, run_id: str, runs: List[Dict[str, Any]]) -> None:
    """Display model metadata and feature information. Args: forecaster (Any): Loaded forecaster model. run_id (str): MLflow run ID. runs (List[Dict[str, Any]]): List of all runs from registry. Returns: None."""
    st.markdown("---")
    st.subheader("Model Information")
    
    current_run = None
    for r in runs:
        if r['run_id'] == run_id:
            current_run = r
            break
    
    with st.expander("Model Metadata", expanded=True):
        if current_run:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Run ID:** `{run_id[:8]}`")
                st.markdown(f"**Training Date:** {current_run['start_time'].strftime('%Y-%m-%d %H:%M')}")
                st.markdown(f"**Model Type:** {current_run.get('tags.model_type', 'XGBoost')}")
            with col2:
                cv_score = current_run.get('metrics.cv_mean_score', 'N/A')
                if cv_score != 'N/A':
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
        if forecaster.ranked_encoders:
            st.markdown("**Ranked Features:**")
            for col_name, encoder in forecaster.ranked_encoders.items():
                levels = list(encoder.mapping.keys())
                st.markdown(f"- **{col_name}**: {len(levels)} levels - {', '.join(levels[:5])}{'...' if len(levels) > 5 else ''}")
        
        if forecaster.proximity_encoders:
            st.markdown("**Proximity Features:**")
            for col_name, encoder in forecaster.proximity_encoders.items():
                st.markdown(f"- **{col_name}**: Proximity-based encoding")
        
        handled = list(forecaster.ranked_encoders.keys()) + list(forecaster.proximity_encoders.keys())
        remaining = [f for f in forecaster.feature_names if f not in handled and f not in [f"{h}_Enc" for h in handled]]
        if remaining:
            st.markdown("**Numerical Features:**")
            st.markdown(f"- {', '.join(remaining[:10])}{'...' if len(remaining) > 10 else ''}")
        
        st.markdown(f"**Total Features:** {len(forecaster.feature_names)}")


def render_inference_ui() -> None:
    """Render the inference interface. Returns: None."""
    st.header("Salary Inference")
    
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
        ts = r['start_time'].strftime('%Y-%m-%d %H:%M')
        m_type = r.get("tags.model_type", "XGBoost") 
        d_name = r.get("tags.dataset_name", "Unknown Data")
        add_tag = r.get("tags.additional_tag")
        if not add_tag or add_tag == "N/A":
             add_tag = r.get("tags.output_filename")
             
        cv_score = fmt_score(r.get('metrics.cv_mean_score', 'N/A'))
        r_id = r['run_id'][:8]
        
        base = f"{ts} | {m_type} | {d_name} | CV:{cv_score} | ID:{r_id}"
        
        if add_tag and add_tag != "N/A":
            base += f" | {add_tag}"
            
        return base

    run_options = {get_run_label(r): r['run_id'] for r in runs}
    
    selected_label = st.selectbox("Select Model Version", options=list(run_options.keys()))
    
    if not selected_label:
        return
        
    run_id = run_options[selected_label]
    
    if "forecaster" not in st.session_state or st.session_state.get("current_run_id") != run_id:
        with st.spinner(f"Loading model from MLflow run {run_id}..."):
            try:
                st.session_state["forecaster"] = registry.load_model(run_id)
                st.session_state["current_run_id"] = run_id
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
    
    forecaster = st.session_state["forecaster"]
    render_model_information(forecaster, run_id, runs)
    
    with st.expander("Model Analysis", expanded=False):
        analytics_service = AnalyticsService()
        targets = analytics_service.get_available_targets(forecaster)
        
        if not targets:
            st.warning("No targets available in this model.")
        else:
            viz_options = ["Feature Importance"]
            selected_viz = st.selectbox(
                "Select Visualization",
                viz_options,
                key="model_analysis_viz"
            )
            
            st.markdown("---")
            
            if selected_viz == "Feature Importance":
                selected_target = st.selectbox(
                    "Select Target Component",
                    targets,
                    key="model_analysis_target"
                )
                
                if selected_target:
                    quantiles = analytics_service.get_available_quantiles(forecaster, selected_target)
                    if quantiles:
                        selected_q_val = st.selectbox(
                            "Select Quantile",
                            quantiles,
                            format_func=lambda x: f"P{int(x*100)}",
                            key="model_analysis_quantile"
                        )
                        
                        df_imp = analytics_service.get_feature_importance(forecaster, selected_target, selected_q_val)
                        if df_imp is not None and not df_imp.empty:
                            st.dataframe(df_imp, width="stretch")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=df_imp.head(20), x="Gain", y="Feature", ax=ax, palette="viridis", hue="Feature", legend=False)
                            ax.set_title(f"Top 20 Features for {selected_target} (P{int(selected_q_val*100)})")
                            st.pyplot(fig)
                        else:
                            st.warning(f"No feature importance scores found for {selected_target} at P{int(selected_q_val*100)}.")
                    else:
                        st.warning(f"No quantiles available for target {selected_target}.")
    
    st.markdown("---")
    
    with st.form("inference_form"):
        st.subheader("Input Features")
        c1, c2 = st.columns(2)
        
        input_data = {}
        
        with c1:
            for col_name, encoder in forecaster.ranked_encoders.items():
                levels = list(encoder.mapping.keys())
                val = st.selectbox(col_name, levels, key=f"input_{col_name}")
                input_data[col_name] = val
            
            for col_name, encoder in forecaster.proximity_encoders.items():
                val = st.text_input(col_name, "New York", key=f"input_{col_name}")
                input_data[col_name] = val
            
        with c2:
            handled = list(forecaster.ranked_encoders.keys()) + list(forecaster.proximity_encoders.keys())
            remaining = [f for f in forecaster.feature_names if f not in handled and f not in [f"{h}_Enc" for h in handled]]
            
            for feat in remaining:
                val = st.number_input(feat, 0, 100, 5, key=f"input_{feat}")
                input_data[feat] = val
            
        submitted = st.form_submit_button("Predict")
    
    if submitted:
        input_df = pd.DataFrame([input_data])
        
        with st.spinner("Predicting..."):
            results = forecaster.predict(input_df)
            
        st.subheader("Prediction Results")
        
        if "Location" in forecaster.proximity_encoders:
             encoder = forecaster.proximity_encoders["Location"]
             loc_val = input_data.get("Location")
             if loc_val:
                 st.markdown(f"**Target Location Zone:** {encoder.mapper.get_zone(loc_val)}")
        
        res_data = []
        for target, preds in results.items():
            row = {"Component": target}
            for q_key, val in preds.items():
                row[q_key] = val[0]
            res_data.append(row)
            
        res_df = pd.DataFrame(res_data)
        chart_df = res_df.set_index("Component").T
        
        try:
            sorted_index = sorted(chart_df.index, key=lambda x: int(x.replace("p", "")))
            chart_df = chart_df.reindex(sorted_index)
        except ValueError:
            pass
            
        st.line_chart(chart_df)
        
        st.dataframe(res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"}))

