import streamlit as st
import pandas as pd
import json
import traceback
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional, Dict, Any, List

# Ensure src can be imported if running from inside src/app or root
current_dir = os.path.dirname(os.path.abspath(__file__))

from src.model.model import SalaryForecaster
from src.app.config_ui import render_config_ui
from src.app.data_analysis import render_data_analysis_ui
from src.app.model_analysis import render_model_analysis_ui
from src.utils.config_loader import get_config
from src.app.caching import load_data_cached as load_data
from src.utils.logger import setup_logging

# Services
from src.services.model_registry import ModelRegistry
from src.services.training_service import TrainingService

def render_inference_ui() -> None:
    """Renders the inference interface."""
    st.header("Salary Inference")
    
    registry = ModelRegistry()
    
    # Check if model is loaded
    # MLflow returns runs, not file paths. 
    # We display: "RunID (Date) - Metric"
    runs = registry.list_models() # List of dicts
    
    if not runs:
        st.warning("No trained models found in MLflow. Please train a new model.")
        return

    # Create display labels
    # Create display labels
    def fmt_score(x):
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)

    run_options = {f"{r['start_time'].strftime('%Y-%m-%d %H:%M')} | CV:{fmt_score(r.get('metrics.cv_mean_score', 'N/A'))} | ID:{r['run_id'][:8]}": r['run_id'] for r in runs}
    
    selected_label = st.selectbox("Select Model Version", options=list(run_options.keys()))
    
    if not selected_label:
        return
        
    run_id = run_options[selected_label]
    
    # Load Model
    if "forecaster" not in st.session_state or st.session_state.get("current_run_id") != run_id:
        with st.spinner(f"Loading model from MLflow run {run_id}..."):
            try:
                st.session_state["forecaster"] = registry.load_model(run_id)
                st.session_state["current_run_id"] = run_id
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return
    
    forecaster: SalaryForecaster = st.session_state["forecaster"]
    
    with st.form("inference_form"):
        st.subheader("Candidate Details")
        c1, c2 = st.columns(2)
        
        with c1:
            level_map = forecaster.level_encoder.mapping
            levels = list(level_map.keys()) if level_map else ["E3", "E4", "E5"]
            level = st.selectbox("Level", levels)
            
            location = st.text_input("Location", "New York")
            
        with c2:
            yoe = st.number_input("Years of Experience", 0, 30, 5)
            yac = st.number_input("Years at Company", 0, 30, 0)
            
        if st.form_submit_button("Predict Compensation"):
            input_df = pd.DataFrame([{
                "Level": level,
                "Location": location,
                "YearsOfExperience": yoe,
                "YearsAtCompany": yac
            }])
            
            with st.spinner("Predicting..."):
                results = forecaster.predict(input_df)
                
            st.subheader("Prediction Results")
            st.markdown(f"**Target Location Zone:** {forecaster.loc_encoder.mapper.get_zone(location)}")
            
            # Prepare data for display
            res_data = []
            for target, preds in results.items():
                row = {"Component": target}
                for q_key, val in preds.items():
                    row[q_key] = val[0]
                res_data.append(row)
                
            res_df = pd.DataFrame(res_data)
            
            # 1. Visualization (Interactive Line Chart)
            # We want X-axis = Percentiles (p10, p25...), Lines = Components
            chart_df = res_df.set_index("Component").T
            
            # Sort index (percentiles) numerically to ensure correct order (e.g. p5 vs p10)
            # Index is currently strings like "p10", "p25"
            try:
                # Extract integer part for sorting
                sorted_index = sorted(chart_df.index, key=lambda x: int(x.replace("p", "")))
                chart_df = chart_df.reindex(sorted_index)
            except ValueError:
                # Fallback if index format is unexpected
                pass
                
            st.line_chart(chart_df)
            
            # 2. Table
            st.dataframe(res_df.style.format({c: "${:,.0f}" for c in res_df.columns if c != "Component"}))

import time

# Service Singleton
@st.cache_resource
def get_training_service() -> TrainingService:
    return TrainingService()

def render_training_ui() -> None:
    """Renders the model training interface."""
    st.header("Model Training")
    
    st.info("Configure settings in 'Configuration' page before training.")
    
    # 1. Shared Data Loading
    df = None
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        st.success(f"Using loaded data from Data Analysis ({len(df)} rows).")
        if st.button("Use Different File"):
            del st.session_state["training_data"]
            st.rerun()
    else:
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state["training_data"] = df # Cache it
                st.success(f"Loaded {len(df)} rows.")
            except Exception as e:
                st.error(f"Error loading file: {e}")
                
    do_tune = st.checkbox("Run Hyperparameter Tuning", value=False)
    num_trials = 20
    if do_tune:
        num_trials = st.number_input("Number of Trials", 5, 100, 20)
        
    remove_outliers = st.checkbox("Remove Outliers (IQR)", value=True)
    
    # Restored: Chart option for post-training display
    display_charts = st.checkbox("Show Training Performance Chart", value=True)
    
    custom_name = st.text_input("Model Output Filename (Optional)", placeholder="e.g. my_custom_model.pkl")
    
    # Initialize Service
    training_service = get_training_service()
    registry = ModelRegistry()

    # Job State Management
    if "training_job_id" not in st.session_state:
        st.session_state["training_job_id"] = None
        
    job_id = st.session_state["training_job_id"]

    # Start Button
    if job_id is None:
        if st.button("Start Training (Async)"):
            if df is None:
                st.error("No data loaded.")
                return
                
            job_id = training_service.start_training_async(
                df, 
                remove_outliers=remove_outliers,
                do_tune=do_tune,
                n_trials=num_trials
            )
            st.session_state["training_job_id"] = job_id
            st.rerun()

    # Polling & Status Display
    else:
        status = training_service.get_job_status(job_id)
        
        if status is None:
            st.error("Job not found. Clearing state.")
            st.session_state["training_job_id"] = None
            st.rerun()
            return

        state = status["status"]
        st.info(f"Training Status: **{state}**")
        
        # Progress Bar / Spinner equivalent
        if state in ["QUEUED", "RUNNING"]:
            with st.spinner("Training in progress... (You can switch tabs, but stay in app to see completion)"):
                # Poll every 2 seconds
                time.sleep(2) 
                st.rerun()
                
        # Show Logs
        with st.expander("Training Logs", expanded=(state != "COMPLETED")):
            st.code("\n".join(status["logs"]))

        # Completion Handling
        if state == "COMPLETED":
            st.success("Training Finished Successfully!")
            
            # --- Result Visualization ---
            history = status.get("history", [])
            results_data = []
            
            for event in history:
                if event.get("stage") == "cv_end":
                    results_data.append({
                        "Model": event.get("model_name"),
                        "Best Round": event.get("best_round"),
                        "Score": event.get("best_score")
                    })
            
            if results_data:
                res_df = pd.DataFrame(results_data)
                if display_charts:
                    st.line_chart(res_df.set_index("Model")["Score"])
                st.dataframe(res_df.style.format({"Score": "{:.4f}"}))
            # ---------------------------
            
            forecaster = status["result"]
            run_id = status.get("run_id", "N/A")
            
            st.session_state["forecaster"] = forecaster
            st.session_state["current_run_id"] = run_id
            
            st.info(f"Model logged to MLflow with Run ID: **{run_id}**")
            st.markdown("[Open MLflow UI](http://localhost:5000) to view details.")
                
            if st.button("Start New Training"):
                st.session_state["training_job_id"] = None
                st.rerun()
                
        elif state == "FAILED":
            st.error(f"Training Failed: {status.get('error')}")
            if st.button("Retry"):
                st.session_state["training_job_id"] = None
                st.rerun()

def main() -> None:
    """Main entry point for the Streamlit application."""
    st.set_page_config(page_title="Salary Forecaster", layout="wide")
    
    # Initialize
    config = get_config()
    st.session_state["config_override"] = config
    
    st.sidebar.title("Navigation")
    
    # Use session state for nav persistence if needed, but sidebar widget handles it
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Inference"
        
    options = ["Inference", "Training", "Data Analysis", "Model Analysis", "Configuration"]
    default_index = 0
    if st.session_state.get("nav") in options:
        default_index = options.index(st.session_state["nav"])
        
    nav = st.sidebar.radio("Go to", options, index=default_index, key="nav_radio")
    
    # Update session state to match
    st.session_state["nav"] = nav
    
    if nav == "Inference":
        render_inference_ui()
    elif nav == "Training":
        render_training_ui()
    elif nav == "Data Analysis":
        render_data_analysis_ui()
    elif nav == "Model Analysis":
        render_model_analysis_ui()
    elif nav == "Configuration":
        new_config = render_config_ui(config)
        st.session_state["config_override"] = new_config

if __name__ == "__main__":
    setup_logging()
    main()
