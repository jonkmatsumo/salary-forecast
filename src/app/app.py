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
from src.utils.config_loader import get_config
from src.utils.data_utils import load_data

st.set_page_config(page_title="Salary Forecast", layout="wide")

st.title("Salary Forecasting Engine")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Inference"])

if page == "Train Model":
    st.header("Train New Model")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Settings")
        model_name = st.text_input("Output Model Name", value="salary_model_web.pkl")
        
        # Determine valid overrides via UI
        current_config = get_config()
        # Render the UI and get the updated config dict
        updated_config = render_config_ui(current_config)
        
    with col2:
        st.subheader("Data & Training")
        uploaded_file = st.file_uploader("Upload Training CSV", type=["csv"])
        
        if uploaded_file is not None:
            # load_data handles cleaning (years parsing, etc.)
            # It accepts the uploaded_file buffer just like pd.read_csv
            df = load_data(uploaded_file)
            st.info(f"Loaded {len(df)} samples.")
            st.dataframe(df.head())
            
            if st.button("Start Training", type="primary"):
                try:
                    # Use the config from the UI
                    config = updated_config
                        
                    success_placeholder = st.empty()
                    status_container = st.status("Training in progress...", expanded=True)
                    
                    # Capture stdout? Streamlit doesn't easily show real-time stdout.
                    # We'll just show status updates.
                    
                    try:
                        status_text = status_container.empty()
                        status_text.markdown("Status: **Initializing model...**")
                        
                        forecaster = SalaryForecaster(config=config)
                        
                        status_text.markdown("Status: **Starting training...**")
                        
                        results_data = []
                        table_placeholder = status_container.empty()
                        
                        def streamlit_callback(msg, data=None):
                            if data and data.get("stage") == "start":
                                # Update current action label
                                status_text.markdown(f"Status: **Training {data['model_name']}...**")
                            elif data and data.get("stage") == "cv_end":
                                 # Append to results
                                 model_name = data.get('model_name', 'Unknown')
                                 # Parse model name (Target_pXX)
                                 if '_p' in model_name:
                                     parts = model_name.rsplit('_', 1)
                                     component = parts[0]
                                     percentile = parts[1] # e.g. p10
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
                                     # Deprecation fix: use_container_width=True -> width="stretch"
                                     table_placeholder.dataframe(df_res, width="stretch")
                            elif data and data.get("stage") == "cv_start":
                                 pass
                        
                        forecaster.train(df, callback=streamlit_callback)
                        
                        status_text.markdown("Status: **Saving model...**")
                        with open(model_name, "wb") as f:
                            pickle.dump(forecaster, f)
                            
                        status_text.markdown("Status: **Completed**")
                        
                        # Show success message at the top
                        success_placeholder.success(f"Model saved as `{model_name}`")
                        
                        # Collapse status to show it as "Details" below
                        status_container.update(label="Training Logic Completed", state="complete", expanded=False)
                        
                    except Exception as e:
                        status_container.update(label="Training Failed", state="error", expanded=True)
                        raise e
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    # Enable traceback for easier debugging

                    st.code(traceback.format_exc())

elif page == "Inference":
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
                
                # Input Form
                # TODO: In a more advanced version, we could inspect `model.feature_names` to build this dynamically.
                # For now, we align with the hardcoded inputs of the project.
                
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
                    
                    # Prepare data for plotting
                    quantiles = sorted(model.quantiles)
                    quantile_labels = [f"P{int(q*100)}" for q in quantiles]
                    
                    # Create Tabs for visual vs table
                    tab1, tab2 = st.tabs(["Visualization", "Data Table"])
                    
                    with tab1:
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
                        
                        # Format Y axis as currency
                        ax.get_yaxis().set_major_formatter(
                            plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
                        )
                        
                        st.pyplot(fig)
                        
                    with tab2:
                        # Build a nice table
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
