import streamlit as st
import pandas as pd
import json
import traceback
import os
import pickle
import glob
import sys
import matplotlib.pyplot as plt

# Ensure src can be imported if running from inside src/app or root
# Assuming we run as `streamlit run src/app/app.py` from root, this might not be needed if root is in PYTHONPATH
# But to be safe let's add the root to path.
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.model.model import SalaryForecaster
from src.app.config_ui import render_config_ui
from src.app.data_analysis import render_data_analysis_ui
from src.app.model_analysis import render_model_analysis_ui
from src.utils.config_loader import get_config
from src.utils.data_utils import load_data

st.set_page_config(page_title="Salary Forecast", layout="wide")

st.title("Salary Forecasting Engine")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Inference", "Train Model", "Data Analysis", "Model Analysis", "Configuration"])

if page == "Inference":
    st.header("Salary Inference")
    
    # Model Selection
    model_files = glob.glob("*.pkl")
    if not model_files:
        st.warning("No model files (*.pkl) found in the root directory. Please train a model first.")
    else:
        selected_model = st.selectbox("Select Model", model_files)
        
        if selected_model:
            try:
                with open(selected_model, "rb") as f:
                    model = pickle.load(f)
                
                st.subheader("Candidate Attributes")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    level = st.selectbox("Level", ["E3", "E4", "E5", "E6", "E7"])
                    location = st.text_input("Location", value="New York")
                with col_b:
                    yoe = st.number_input("Years of Experience", min_value=0, value=3, step=1)
                    yac = st.number_input("Years at Company", min_value=0, value=0, step=1)
                
                if st.button("Generate Forecast", type="primary"):
                    input_df = pd.DataFrame([{
                        "Level": level,
                        "Location": location,
                        "YearsOfExperience": yoe,
                        "YearsAtCompany": yac
                    }])
                    
                    prediction = model.predict(input_df)
                    
                    st.markdown("### Results")
                    
                    quantiles = sorted(model.quantiles)
                    quantile_labels = [f"P{int(q*100)}" for q in quantiles]
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    for target, preds in prediction.items():
                        y_vals = [preds[f"p{int(q*100)}"][0] for q in quantiles]
                        ax.plot(quantiles, y_vals, marker='o', label=target)
                        
                    ax.set_title("Predicted Compensation Distribution")
                    ax.set_xlabel("Quantile")
                    ax.set_ylabel("Amount ($)")
                    ax.set_xticks(quantiles)
                    ax.set_xticklabels(quantile_labels)
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.7)
                    
                    ax.get_yaxis().set_major_formatter(
                        plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
                    )
                    
                    st.pyplot(fig)
                    
                    st.subheader("Detailed Data")
                    table_data = []
                    for target, preds in prediction.items():
                        row = {"Component": target}
                        for q in quantiles:
                            val = preds[f"p{int(q*100)}"][0]
                            row[f"P{int(q*100)}"] = f"${val:,.0f}"
                        table_data.append(row)
                    
                    st.table(pd.DataFrame(table_data))
                        
            except Exception as e:
                st.error(f"Error loading or running model: {e}")
                st.code(traceback.format_exc())


elif page == "Train Model":
    st.header("Train New Model")
    
    if "config_override" in st.session_state:
        config = st.session_state["config_override"]
        st.success("Using custom configuration.")
        with st.expander("View Active Config"):
            st.json(config)
    else:
        config = get_config()
        st.info("Using default configuration.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        model_name = st.text_input("Output Model Name", value="salary_model_web.pkl")
        
        do_tune = st.checkbox("Enable Auto-Tuning (Optuna)", value=False, help="Optimize hyperparameters using Bayesian search (slower).")
        if do_tune:
            num_trials = st.number_input("Number of Trials", min_value=5, value=20, step=5)
        else:
            num_trials = 20
        
        remove_outliers = st.checkbox("Remove Outliers (IQR)", value=False, help="Filter data using Interquartile Range.")
        
    with col2:
        st.subheader("Data & Training")
        
        # Check for shared data
        if "training_data" in st.session_state:
            st.info(f"Using loaded data ({len(st.session_state['training_data'])} rows).")
            if st.button("Use Different Data"):
                del st.session_state["training_data"]
                st.rerun()
            df = st.session_state["training_data"]
        else:
            uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
            if uploaded_file is not None:
                try:
                    df = load_data(uploaded_file)
                    st.session_state["training_data"] = df
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading CSV: {e}")
                    df = None
            else:
                df = None
        
        if df is not None:
             # Validation
             required_cols = ["Level", "Location", "YearsOfExperience", "YearsAtCompany", "Date", "BaseSalary", "Stock", "Bonus", "TotalComp"]
             missing_cols = [c for c in required_cols if c not in df.columns]
             
             if missing_cols:
                 st.error(f"uploaded CSV is missing required columns: {', '.join(missing_cols)}")
             else:
                 st.dataframe(df.head())
                 
                 if st.button("Start Training", type="primary"):
                     try:
                         success_placeholder = st.empty()
                         status_container = st.status("Training in progress...", expanded=True)
                         
                         try:
                             status_text = status_container.empty()
                             status_text.markdown("Status: **Initializing model...**")
                             
                             forecaster = SalaryForecaster(config=config)
                             
                             if do_tune:
                                 status_text.markdown(f"Status: **Tuning hyperparameters ({num_trials} trials)...**")
                                 with st.spinner(f"Running Optuna optimization ({num_trials} trials)..."):
                                     best_params = forecaster.tune(df, n_trials=num_trials)
                                     st.write("Best Hyperparameters:", best_params)
                             
                             status_text.markdown("Status: **Starting training...**")
                             
                             results_data = []
                             table_placeholder = status_container.empty()
                             
                             def streamlit_callback(msg, data=None):
                                 if data and data.get("stage") == "start":
                                     status_text.markdown(f"Status: **Training {data['model_name']}...**")
                                 elif data and data.get("stage") == "cv_end":
                                      model_name = data.get('model_name', 'Unknown')
                                      if '_p' in model_name:
                                          parts = model_name.rsplit('_', 1)
                                          component = parts[0]
                                          percentile = parts[1]
                                      else:
                                          component = model_name
                                          percentile = "-"
     
                                      results_data.append({
                                          "Component": component,
                                          "Percentile": percentile,
                                          "Best Round": data.get('best_round'),
                                          "Metric": data.get('metric_name'),
                                          "Score": f"{data.get('best_score'):.4f}"
                                      })
     
                                      if results_data:
                                          df_res = pd.DataFrame(results_data)
                                          table_placeholder.dataframe(df_res, width="stretch")
                             
                             forecaster.train(df, callback=streamlit_callback, remove_outliers=remove_outliers)
                             
                             status_text.markdown("Status: **Saving model...**")
                             with open(model_name, "wb") as f:
                                 pickle.dump(forecaster, f)
                                 
                             status_text.markdown("Status: **Completed**")
                             success_placeholder.success(f"Model saved as `{model_name}`")
                             status_container.update(label="Training Logic Completed", state="complete", expanded=False)
                             
                         except Exception as e:
                             status_container.update(label="Training Failed", state="error", expanded=True)
                             raise e
                         
                     except Exception as e:
                         st.error(f"Training failed: {e}")
                         st.code(traceback.format_exc())

elif page == "Data Analysis":
    render_data_analysis_ui()

elif page == "Model Analysis":
    render_model_analysis_ui()

elif page == "Configuration":
    st.header("Configuration")
    current_config = get_config()
    updated_config = render_config_ui(current_config)
    st.session_state["config_override"] = updated_config
