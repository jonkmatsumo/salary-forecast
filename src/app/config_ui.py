import streamlit as st
import pandas as pd
import json
import copy

def render_save_load_controls(current_config_state):
    """
    Renders Save/Load controls.
    current_config_state: The configuration dictionary currently being edited/displayed (to be saved).
    """
    st.markdown("---")
    st.subheader("Config Management")
    
    # Save
    config_json = json.dumps(current_config_state, indent=2)
    st.download_button(
        label="Download Config JSON",
        data=config_json,
        file_name="config.json",
        mime="application/json"
    )
    
    # Load
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


def render_levels_editor(config):
    """
    Renders an editor for the 'mappings.levels' section.
    Returns the updated levels dictionary.
    """
    st.subheader("Levels Configuration")
    
    levels_dict = config.get("mappings", {}).get("levels", {})
    
    data = [{"Level": k, "Rank": v} for k, v in levels_dict.items()]
    df = pd.DataFrame(data)
    
    # Editable dataframe
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        # Deprecation fix: use_container_width=True -> width="stretch"
        width="stretch",
        key="levels_editor",
        column_config={
            "Level": st.column_config.TextColumn("Level Name", required=True),
            "Rank": st.column_config.NumberColumn("Rank Value", required=True, min_value=0, step=1)
        }
    )
    
    # Reconstruct dictionary
    new_levels = {}
    for index, row in edited_df.iterrows():
        if row["Level"]: # Ensure not empty
            new_levels[row["Level"]] = int(row["Rank"])
            
    return new_levels

def render_location_targets_editor(config):
    """
    Renders an editor for 'mappings.location_targets'.
    Returns updated location_targets dictionary.
    """
    st.subheader("Location Targets")
    
    loc_dict = config.get("mappings", {}).get("location_targets", {})
    
    data = [{"City": k, "Tier/Rank": v} for k, v in loc_dict.items()]
    df = pd.DataFrame(data)
    
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        # Deprecation fix: use_container_width=True -> width="stretch"
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

def render_location_settings_editor(config):
    """
    Renders slider for location settings.
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

def render_model_config_editor(config):
    """
    Renders editor for 'model' configuration.
    """
    st.subheader("Model Configuration")
    
    model_config = config.get("model", {})
    
    # 1. Targets
    defaults_targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    current_targets = model_config.get("targets", defaults_targets)
    
    st.markdown("**Targets**")
    targets_df = pd.DataFrame([{"Target": t} for t in current_targets])
    edited_targets_df = st.data_editor(
        targets_df,
        num_rows="dynamic",
        width="stretch",
        key="targets_editor",
        column_config={
            "Target": st.column_config.TextColumn("Target Variable", required=True)
        }
    )
    new_targets = [row["Target"] for _, row in edited_targets_df.iterrows() if row["Target"]]
    
    # 2. Quantiles
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
    
    # 3. Sample Weight
    default_k = 1.0
    current_k = model_config.get("sample_weight_k", default_k)
    new_k = st.number_input(
        "Sample Weight K", 
        value=float(current_k), 
        min_value=0.0, 
        step=0.1,
        help="Controls how much recent data is prioritized (higher = more weight to recent data)."
    )
    
    # 4. Hyperparameters
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

    # 5. Features (Monotone Constraints)
    st.markdown("**Features & Constraints**")
    default_features = [
        {"name": "Level_Enc", "monotone_constraint": 1},
        {"name": "Location_Enc", "monotone_constraint": 0},
        {"name": "YearsOfExperience", "monotone_constraint": 1},
        {"name": "YearsAtCompany", "monotone_constraint": 0}
    ]
    current_features = model_config.get("features", default_features)
    
    features_df = pd.DataFrame(current_features)
    edited_features_df = st.data_editor(
        features_df,
        num_rows="dynamic",
        width="stretch",
        key="features_editor",
        column_config={
            "name": st.column_config.TextColumn("Feature Name", required=True),
            "monotone_constraint": st.column_config.NumberColumn("Monotone Constraint (-1, 0, 1)", min_value=-1, max_value=1, step=1)
        }
    )
    new_features = edited_features_df.to_dict(orient="records")
    
    return {
        "targets": new_targets,
        "quantiles": new_quantiles,
        "sample_weight_k": new_k,
        "hyperparameters": new_hyperparams,
        "features": new_features
    }

def render_config_ui(config):
    """
    Main entry point to render the full config UI.
    Returns: A NEW config dictionary with updates applied.
    """
    # Check if there is an override from the loader
    if "config_override" in st.session_state:
         # Use the loaded config as the base instead of the file/default
         config = st.session_state["config_override"]
    
    new_config = copy.deepcopy(config)
    
    # Ensure structure exists
    if "mappings" not in new_config:
        new_config["mappings"] = {}
        
    # Levels
    new_config["mappings"]["levels"] = render_levels_editor(new_config)
    
    # Locations
    new_config["mappings"]["location_targets"] = render_location_targets_editor(new_config)
    
    # Settings
    new_config["location_settings"] = render_location_settings_editor(new_config)

    # Model
    new_config["model"] = render_model_config_editor(new_config)
    
    # Save/Load Controls
    render_save_load_controls(new_config)
    
    return new_config
