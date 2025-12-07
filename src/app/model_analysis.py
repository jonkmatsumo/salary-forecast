import streamlit as st
import pandas as pd
import glob
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

def render_model_analysis_ui() -> None:
    st.header("Model Analysis")
    
    # 1. Model Selection
    model_files = glob.glob("*.pkl")
    if not model_files:
        st.warning("No model files (*.pkl) found in the root directory. Please train a model first.")
        return

    selected_model_file = st.selectbox("Select Model to Analyze", model_files)
    
    if selected_model_file:
        try:
            with open(selected_model_file, "rb") as f:
                forecaster = pickle.load(f)
                
            st.success(f"Loaded `{selected_model_file}`")
            
            # 2. Inspector
            st.subheader("Feature Importance")
            st.info("Visualize which features drive the predictions (Gain metric).")
            
            # Get available targets and quantiles from the loaded forecaster
            # forecaster.models is a dict of {target: {quantile_str: model}}
            if not hasattr(forecaster, "models") or not forecaster.models:
                st.error("This model file does not appear to contain trained models.")
                return
                
            # Forecaster stores models as flat dict: "Target_p50": model
            # We can use metadata attributes if available, or parse keys
            
            if hasattr(forecaster, "targets"):
                targets = forecaster.targets
            else:
                # Fallback: Parse from keys
                keys = list(forecaster.models.keys())
                targets = sorted(list(set([k.split('_p')[0] for k in keys if '_p' in k])))

            selected_target = st.selectbox("Select Target Component", targets)
            
            if selected_target:
                # Find available quantiles for this target
                if hasattr(forecaster, "quantiles"):
                     quantiles = sorted(forecaster.quantiles)
                else:
                     # Fallback
                     keys = list(forecaster.models.keys())
                     target_keys = [k for k in keys if k.startswith(f"{selected_target}_p")]
                     # Extract XX from pXX
                     quantiles = sorted([float(k.split('_p')[1])/100 for k in target_keys])

                selected_q_val = st.selectbox("Select Quantile", quantiles, format_func=lambda x: f"P{int(x*100)}")
                
                if selected_q_val is not None:
                    model_name = f"{selected_target}_p{int(selected_q_val*100)}"
                    
                    if model_name in forecaster.models:
                        model = forecaster.models[model_name]
                    
                        # Extract importance
                        # XGBoost sklearn API: model.get_booster().get_score(importance_type='gain')
                        # Or check if it's a raw booster or sklearn wrapper
                        try:
                            if hasattr(model, "get_booster"):
                                booster = model.get_booster()
                            else:
                                # It might be the booster itself
                                booster = model
                            
                            importance = booster.get_score(importance_type="gain")
                            
                            if not importance:
                                st.warning("No feature importance scores found (model might be constant or trained on single feature?).")
                            else:
                                # Convert to DataFrame
                                df_imp = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"])
                                df_imp = df_imp.sort_values(by="Gain", ascending=False)
                                
                                # Plot
                                fig, ax = plt.subplots(figsize=(10, 6))
                                sns.barplot(data=df_imp, x="Gain", y="Feature", hue="Feature", ax=ax, palette="viridis", legend=False)
                                ax.set_title(f"Feature Importance (Gain) - {selected_target} P{int(selected_q_val*100)}")
                                st.pyplot(fig)
                                
                                with st.expander("View Raw Scores"):
                                    st.dataframe(df_imp)
                                    
                        except Exception as e:
                            st.error(f"Failed to extract feature importance: {e}")
                    else:
                        st.error(f"Model keys for {model_name} not found in pickle file.")
                        
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.code(traceback.format_exc())
