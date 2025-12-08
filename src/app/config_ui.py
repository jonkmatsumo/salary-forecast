import streamlit as st
import pandas as pd
import json
import copy
from typing import Dict, Any, List
from src.services.config_generator import ConfigGenerator
from src.app.caching import load_data_cached
from src.utils.csv_validator import validate_csv

def render_save_load_controls(current_config_state: Dict[str, Any]) -> None:
    """Renders Save/Load controls.

    Args:
        current_config_state (Dict[str, Any]): The configuration dictionary currently being edited/displayed.
    """
    st.markdown("---")
    st.subheader("Config Management")
    
    config_json = json.dumps(current_config_state, indent=2)
    st.download_button(
        label="Download Config JSON",
        data=config_json,
        file_name="config.json",
        mime="application/json"
    )
    
    uploaded_file = st.file_uploader("Load Config JSON", type=["json"], key="config_loader")
    if uploaded_file is not None:
        try:
            loaded_config = json.load(uploaded_file)

            if st.session_state.get('loaded_config_content') != loaded_config:
                st.session_state['loaded_config_content'] = loaded_config
                st.session_state['config_override'] = loaded_config
                st.rerun()
                
        except json.JSONDecodeError:
            st.error("Invalid JSON file.")


def render_levels_editor(config: Dict[str, Any]) -> Dict[str, int]:
    """Renders an editor for the 'mappings.levels' section.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.

    Returns:
        Dict[str, int]: Updated levels dictionary.
    """
    st.subheader("Levels Configuration")
    
    levels_dict = config.get("mappings", {}).get("levels", {})
    
    data = [{"Level": k, "Rank": v} for k, v in levels_dict.items()]
    df = pd.DataFrame(data)
    
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key="levels_editor",
        column_config={
            "Level": st.column_config.TextColumn("Level Name", required=True),
            "Rank": st.column_config.NumberColumn("Rank Value", required=True, min_value=0, step=1)
        }
    )
    
    new_levels = {}
    for index, row in edited_df.iterrows():
        if row["Level"]: 
            new_levels[row["Level"]] = int(row["Rank"])
            
    return new_levels

def render_location_targets_editor(config: Dict[str, Any]) -> Dict[str, int]:
    """Renders an editor for 'mappings.location_targets'.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.

    Returns:
        Dict[str, int]: Updated location_targets dictionary.
    """
    st.subheader("Location Targets")
    
    loc_dict = config.get("mappings", {}).get("location_targets", {})
    
    data = [{"City": k, "Tier/Rank": v} for k, v in loc_dict.items()]
    df = pd.DataFrame(data)
    
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key="locations_editor",
        column_config={
            "City": st.column_config.TextColumn("City, State", required=True),
            "Tier/Rank": st.column_config.NumberColumn("Tier (Rank)", required=True, min_value=1, step=1)
        }
    )
    
    new_locs = {}
    for index, row in edited_df.iterrows():
        if row["City"]:
            new_locs[row["City"]] = int(row["Tier/Rank"])
            
    return new_locs

def render_location_settings_editor(config: Dict[str, Any]) -> Dict[str, Any]:
    """Renders slider for location settings.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.

    Returns:
        Dict[str, Any]: Updated location_settings dictionary.
    """
    st.subheader("Location Settings")
    
    loc_settings = config.get("location_settings", {})
    current_dist = loc_settings.get("max_distance_km", 50)
    
    new_dist = st.slider(
        "Max Distance (km) for Proximity Matching",
        min_value=0,
        max_value=200,
        value=current_dist,
        step=5,
        help="Maximum distance in km to consider a candidate 'local' to a target city."
    )
    
    return {"max_distance_km": new_dist}

def render_model_config_editor(config: Dict[str, Any]) -> Dict[str, Any]:
    """Renders editor for 'model' configuration.

    Args:
        config (Dict[str, Any]): Full configuration dictionary.

    Returns:
        Dict[str, Any]: Updated model configuration dictionary.
    """
    st.subheader("Model Configuration")
    
    model_config = config.get("model", {})
    

    # Combined Variables Table
    st.markdown("**Model Variables (Features & Targets)**")
    
    # 1. Flatten current config into a unified list
    defaults_targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    current_targets = model_config.get("targets", defaults_targets)
    
    default_features = [
        {"name": "Level_Enc", "monotone_constraint": 1},
        {"name": "Location_Enc", "monotone_constraint": 0},
        {"name": "YearsOfExperience", "monotone_constraint": 1},
        {"name": "YearsAtCompany", "monotone_constraint": 0}
    ]
    current_features = model_config.get("features", default_features)
    
    # Create unified dataframe
    # Schema: Name, Role, Monotone Constraint
    unified_data = []
    
    for t in current_targets:
        unified_data.append({
            "Name": t,
            "Role": "Target",
            "Monotone Constraint": 0 # Default/Irrelevant for targets but good to have schema consistency or hide it
        })
        
    for f in current_features:
        unified_data.append({
            "Name": f.get("name"),
            "Role": "Feature",
            "Monotone Constraint": f.get("monotone_constraint", 0)
        })
        
    unified_df = pd.DataFrame(unified_data)
    
    # Render Editor
    edited_vars_df = st.data_editor(
        unified_df,
        num_rows="dynamic",
        width="stretch",
        key="variables_editor",
        column_config={
            "Name": st.column_config.TextColumn("Variable Name", required=True),
            "Role": st.column_config.SelectboxColumn("Role", options=["Feature", "Target", "Ignore"], required=True),
            "Monotone Constraint": st.column_config.NumberColumn(
                "Monotone Constraint (-1, 0, 1)", 
                min_value=-1, 
                max_value=1, 
                step=1,
                help="Constraint for Features: 1 (increasing), -1 (decreasing), 0 (none). Ignored for Targets."
            )
        }
    )
    
    # 2. Parse back into separate lists
    new_targets = []
    new_features = []
    
    for _, row in edited_vars_df.iterrows():
        name = row["Name"]
        role = row["Role"]
        constraint = int(row["Monotone Constraint"])
        
        if not name or role == "Ignore":
            continue
            
        if role == "Target":
            new_targets.append(name)
        elif role == "Feature":
            new_features.append({
                "name": name,
                "monotone_constraint": constraint
            })
    
    # Quantiles configuration (kept separate as it's global model setting)
    defaults_quantiles = [0.1, 0.25, 0.50, 0.75, 0.9]
    current_quantiles = model_config.get("quantiles", defaults_quantiles)
    
    st.markdown("**Quantiles**")
    quantiles_df = pd.DataFrame([{"Quantile": q} for q in current_quantiles])
    edited_quantiles_df = st.data_editor(
        quantiles_df,
        num_rows="dynamic",
        width="stretch",
        key="quantiles_editor",
        column_config={
            "Quantile": st.column_config.NumberColumn("Quantile (0.0-1.0)", min_value=0.0, max_value=1.0, required=True)
        }
    )
    new_quantiles = [float(row["Quantile"]) for _, row in edited_quantiles_df.iterrows()]
    
    # Sample Weight configuration
    default_k = 1.0
    current_k = model_config.get("sample_weight_k", default_k)
    new_k = st.number_input(
        "Sample Weight K", 
        value=float(current_k), 
        min_value=0.0, 
        step=0.1,
        help="Controls how much recent data is prioritized (higher = more weight to recent data)."
    )
    
    st.markdown("**Hyperparameters**")
    default_hyperparams = {
        "training": {"objective": "reg:quantileerror", "tree_method": "hist", "verbosity": 0}, 
        "cv": {"num_boost_round": 100, "nfold": 5, "early_stopping_rounds": 10, "verbose_eval": False}
    }
    current_hyperparams = model_config.get("hyperparameters", default_hyperparams)
    
    t_col, cv_col = st.columns(2)
    with t_col:
        st.write("Training")
        hp_train = current_hyperparams.get("training", {})
        obj = st.text_input("Objective", value=hp_train.get("objective", "reg:quantileerror"), help="XGBoost objective function.")
        tm = st.selectbox("Tree Method", ["hist", "auto", "exact", "approx"], index=0 if hp_train.get("tree_method") == "hist" else 0, help="Tree construction algorithm.") 
        verb = st.number_input("Verbosity", value=hp_train.get("verbosity", 0), help="0 (silent), 1 (warning), 2 (info), 3 (debug)")
        
    with cv_col:
        st.write("Cross-Validation")
        hp_cv = current_hyperparams.get("cv", {})
        nbr = st.number_input("Num Boost Rounds", value=hp_cv.get("num_boost_round", 100), help="Number of boosting iterations.")
        nfold = st.number_input("N Folds", value=hp_cv.get("nfold", 5), help="Number of cross-validation folds.")
        esr = st.number_input("Early Stopping Rounds", value=hp_cv.get("early_stopping_rounds", 10), help="Stop if validation score doesn't improve for N rounds.")
    
    new_hyperparams = {
        "training": {"objective": obj, "tree_method": tm, "verbosity": int(verb)},
        "cv": {"num_boost_round": int(nbr), "nfold": int(nfold), "early_stopping_rounds": int(esr), "verbose_eval": False}
    }

    
    return {
        "targets": new_targets,
        "quantiles": new_quantiles,
        "sample_weight_k": new_k,
        "hyperparameters": new_hyperparams,
        "features": new_features
    }

def render_config_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main entry point to render the full config UI.

    Args:
        config (Dict[str, Any]): Full configuration dictionary to edit.

    Returns:
        Dict[str, Any]: A NEW config dictionary with updates applied.
    """
    st.header("Configuration")
    
    
    # --- Config Generator Section ---
    
    with st.expander("Generate Configuration from Data", expanded=False):
        st.write("Automatically generate a configuration by analyzing your dataset.")
        

        data_source = st.radio("Data Source", ["Use Loaded Training Data", "Upload New CSV"], horizontal=True)
        
        df_to_analyze = None
        
        if data_source == "Use Loaded Training Data":
            if "training_data" in st.session_state:
                df_to_analyze = st.session_state["training_data"]
                st.success(f"Using loaded data ({len(df_to_analyze)} rows).")
            else:
                st.warning("No training data loaded. Please upload a CSV.")
        else:
            uploaded_file = st.file_uploader("Upload CSV Sample", type=["csv"], key="config_gen_uploader")
            if uploaded_file:
                is_valid, err, df = validate_csv(uploaded_file)
                if is_valid:
                    df_to_analyze = df
                    st.success(f"CSV Validated ({len(df)} rows).")
                else:
                    st.error(f"Invalid CSV: {err}")


        use_ai = st.checkbox("Use AI (LLM)", value=True, help="Use Large Language Model to infer configuration. Uncheck to use simple heuristics.")
        
        preset = "none"
        provider = "openai"
        
        if use_ai:
            col_p, col_pre = st.columns(2)
            with col_p:
                provider = st.selectbox("LLM Provider", ["openai", "gemini"], index=0)
            with col_pre:
                preset = st.selectbox("Preset (Domain)", ["none", "salary"], index=1, format_func=lambda x: "Generic" if x == "none" else "Salary Forecasting")
        
        if st.button("Generate Configuration", disabled=(df_to_analyze is None), type="primary"):
            generator = ConfigGenerator()
            try:
                with st.spinner("Analyzing data and generating configuration..."):
                    new_proposal = generator.generate_config(
                        df_to_analyze, 
                        use_llm=use_ai, 
                        provider=provider, 
                        preset=preset
                    )
                    
                st.session_state["config_override"] = new_proposal
                st.success("Configuration generated successfully! Review the proposal below.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
                if use_ai:
                    st.info("Tip: You can try unchecking 'Use AI' to use basic heuristics instead.")

    st.markdown("---")
    # -------------------------------

    if "config_override" in st.session_state:
         config = st.session_state["config_override"]
    
    new_config = copy.deepcopy(config)
    
    if "mappings" not in new_config:
        new_config["mappings"] = {}
        
    new_config["mappings"]["levels"] = render_levels_editor(new_config)
    new_config["mappings"]["location_targets"] = render_location_targets_editor(new_config)
    new_config["location_settings"] = render_location_settings_editor(new_config)
    new_config["model"] = render_model_config_editor(new_config)
    
    render_save_load_controls(new_config)
    
    return new_config
