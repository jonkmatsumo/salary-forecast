"""
Configuration UI for the Streamlit app.

This module provides the UI for the multi-step agentic configuration workflow:
1. Column Classification - Identify targets, features, and columns to ignore
2. Feature Encoding - Determine how to encode categorical features
3. Model Configuration - Configure hyperparameters and constraints
"""

import streamlit as st
import pandas as pd
import json
import copy
from typing import Dict, Any, Optional
import pandas.api.types as pd_types

from src.services.workflow_service import WorkflowService, get_workflow_providers
from src.app.caching import load_data_cached
from src.utils.csv_validator import validate_csv


def _get_progress_message(service: Optional[WorkflowService]) -> str:
    """Get progress message based on current node. Args: service (Optional[WorkflowService]): Workflow service. Returns: str: Progress message."""
    if not service or not service.workflow:
        return "Processing..."
    
    current_node = service.workflow.current_state.get("current_node")
    if not current_node:
        return "Processing..."
    
    node_messages = {
        "validating_input": "Validating input...",
        "classifying_columns": "Classifying columns...",
        "evaluating_features": "Evaluating features...",
        "configuring_model": "Configuring model...",
        "building_config": "Building final configuration..."
    }
    
    return node_messages.get(current_node, "Processing...")

def render_workflow_wizard(df: pd.DataFrame, provider: str = "openai") -> Optional[Dict[str, Any]]:
    """Render the multi-step agentic workflow wizard. Args: df (pd.DataFrame): DataFrame to analyze. provider (str): LLM provider. Returns: Optional[Dict[str, Any]]: Final configuration if complete, None otherwise."""
    if "workflow_service" not in st.session_state:
        st.session_state["workflow_service"] = None
    if "workflow_phase" not in st.session_state:
        st.session_state["workflow_phase"] = "not_started"
    if "workflow_result" not in st.session_state:
        st.session_state["workflow_result"] = None
    
    # Display current phase indicator
    phases = ["Column Classification", "Feature Encoding", "Model Configuration"]
    current_phase_idx = {
        "not_started": -1,
        "classification": 0,
        "encoding": 1,
        "configuration": 2,
        "complete": 3
    }.get(st.session_state["workflow_phase"], -1)
    
    # Phase indicator
    cols = st.columns(3)
    for i, phase in enumerate(phases):
        with cols[i]:
            if i < current_phase_idx:
                st.success(f"Step {i+1}: {phase}")
            elif i == current_phase_idx:
                st.info(f"Step {i+1}: {phase}")
            else:
                st.write(f"Step {i+1}: {phase}")
    
    st.markdown("---")
    
    if st.session_state["workflow_phase"] == "not_started":
        preset_options = ["None", "salary"]
        preset_key = "workflow_preset_selector"
        if preset_key not in st.session_state:
            st.session_state[preset_key] = "None"
        
        selected_preset = st.selectbox(
            "Optional Preset Prompt (for domain-specific guidance)",
            preset_options,
            index=preset_options.index(st.session_state[preset_key]),
            key=preset_key,
            help="Select a preset prompt to provide domain-specific guidance to the AI agents"
        )
        
        preset_value = None if selected_preset == "None" else selected_preset.lower()
        
        if st.button("Start AI-Powered Configuration Wizard", type="primary"):
            service = WorkflowService(provider=provider)
            st.session_state["workflow_service"] = service
            try:
                # Use status container for better progress visibility
                with st.status("Initializing workflow...", expanded=False) as status:
                    status.update(label="Validating input...", state="running")
                    result = service.start_workflow(df, preset=preset_value)
                    status.update(label="Workflow initialized", state="complete")
                st.session_state["workflow_result"] = result
                st.session_state["workflow_phase"] = "classification"
                st.rerun()
            except Exception as e:
                st.error(f"Failed to start workflow: {e}")
        return None
    
    service: WorkflowService = st.session_state["workflow_service"]
    result = st.session_state.get("workflow_result")
    
    if result is None:
        return None
    
    if result.get("status") == "error":
        st.error(f"Workflow error: {result.get('error')}")
        if st.button("Restart Workflow"):
            _reset_workflow_state()
            st.rerun()
        return None
    
    if st.session_state["workflow_phase"] == "classification":
        return _render_classification_phase(service, result, df)
    elif st.session_state["workflow_phase"] == "encoding":
        return _render_encoding_phase(service, result)
    elif st.session_state["workflow_phase"] == "configuration":
        return _render_configuration_phase(service, result)
    elif st.session_state["workflow_phase"] == "complete":
        return _render_complete_phase(result)
    
    return None


def _render_classification_phase(service: WorkflowService, result: Dict[str, Any], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Render the column classification review phase. Args: service (WorkflowService): Workflow service. result (Dict[str, Any]): Classification result. df (pd.DataFrame): Data. Returns: Optional[Dict[str, Any]]: Config if confirmed, None otherwise."""
    st.subheader("Step 1: Column Classification")
    
    data = result.get("data", {})
    reasoning = data.get("reasoning", "")
    if reasoning:
        with st.expander("Agent Reasoning", expanded=True):
            st.markdown(reasoning)
    
    # Editable classification tables
    st.markdown("**Review and edit the column classifications:**")
    
    all_columns = df.columns.tolist()
    targets = data.get("targets", [])
    features = data.get("features", [])
    ignore = data.get("ignore", [])
    
    # Get column_types from workflow state
    column_types = service.workflow.current_state.get("column_types", {}) if service.workflow else {}
    
    # Build unified editor
    classification_data = []
    for col in all_columns:
        if col in targets:
            role = "Target"
        elif col in features:
            role = "Feature"
        elif col in ignore:
            role = "Ignore"
        else:
            role = "Unclassified"
        
        # Format dtype to show semantic type if available
        dtype_display = str(df[col].dtype)
        if col in column_types:
            semantic_type = column_types[col]
            if semantic_type == "location":
                dtype_display = f"string (location)"
            elif semantic_type == "datetime":
                dtype_display = f"{dtype_display} (datetime)"
        
        classification_data.append({"Column": col, "Role": role, "Dtype": dtype_display})
    
    class_df = pd.DataFrame(classification_data)
    
    edited_df = st.data_editor(
        class_df,
        key="classification_editor",
        column_config={
            "Column": st.column_config.TextColumn("Column Name", disabled=True),
            "Role": st.column_config.SelectboxColumn(
                "Classification",
                options=["Target", "Feature", "Ignore", "Unclassified"],
                required=True
            ),
            "Dtype": st.column_config.TextColumn("Data Type", disabled=True)
        },
        hide_index=True,
        width='stretch'
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Confirm & Continue", type="primary"):
            # Parse edited classification
            new_targets = []
            new_features = []
            new_ignore = []
            
            for _, row in edited_df.iterrows():
                col_name = row["Column"]
                role = row["Role"]
                if role == "Target":
                    new_targets.append(col_name)
                elif role == "Feature":
                    new_features.append(col_name)
                elif role == "Ignore":
                    new_ignore.append(col_name)
            
            modifications = {
                "targets": new_targets,
                "features": new_features,
                "ignore": new_ignore
            }
            
            progress_msg = _get_progress_message(service)
            with st.status(progress_msg, expanded=False) as status:
                status.update(label="Evaluating features...", state="running")
                result = service.confirm_classification(modifications)
                status.update(label="Feature encoding complete", state="complete")
            st.session_state["workflow_result"] = result
            st.session_state["workflow_phase"] = "encoding"
            st.rerun()
    
    with col2:
        if st.button("Reset Workflow"):
            _reset_workflow_state()
            st.rerun()
    
    return None


def _render_encoding_phase(service: WorkflowService, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Render the feature encoding review phase. Args: service (WorkflowService): Workflow service. result (Dict[str, Any]): Encoding result. Returns: Optional[Dict[str, Any]]: Config if confirmed, None otherwise."""
    st.subheader("Step 2: Feature Encoding")
    
    data = result.get("data", {})
    
    # Show summary
    summary = data.get("summary", "")
    if summary:
        with st.expander("Agent Summary", expanded=True):
            st.markdown(summary)
    
    # Editable encoding table
    st.markdown("**Review and edit encoding strategies:**")
    
    encodings = data.get("encodings", {})
    
    encoding_data = []
    for col, config in encodings.items():
        enc_type = config.get("type", "unknown")
        mapping = config.get("mapping", {})
        reasoning = config.get("reasoning", "")
        
        encoding_data.append({
            "Column": col,
            "Encoding": enc_type,
            "Mapping": json.dumps(mapping) if mapping else "",
            "Notes": reasoning
        })
    
    if not encoding_data:
        st.info("No features require encoding (all numeric).")
        encoding_data = [{"Column": "", "Encoding": "numeric", "Mapping": "", "Notes": ""}]
    
    enc_df = pd.DataFrame(encoding_data)
    
    edited_enc_df = st.data_editor(
        enc_df,
        key="encoding_editor",
        column_config={
            "Column": st.column_config.TextColumn("Feature Column"),
            "Encoding": st.column_config.SelectboxColumn(
                "Encoding Type",
                options=["numeric", "ordinal", "onehot", "proximity", "label"],
                required=True
            ),
            "Mapping": st.column_config.TextColumn(
                "Mapping (JSON)",
                help="For ordinal encoding, provide a JSON mapping like {\"Low\": 0, \"High\": 1}"
            ),
            "Notes": st.column_config.TextColumn("Notes")
        },
        hide_index=True,
        width='stretch',
        num_rows="dynamic"
    )
    
    # Mapping editor for ordinal columns
    ordinal_cols = [row["Column"] for _, row in edited_enc_df.iterrows() 
                    if row["Encoding"] == "ordinal" and row["Column"]]
    
    if ordinal_cols:
        st.markdown("**Edit Ordinal Mappings:**")
        for col in ordinal_cols:
            with st.expander(f"Mapping for: {col}"):
                # Get current mapping
                current_mapping = {}
                for _, row in edited_enc_df.iterrows():
                    if row["Column"] == col and row["Mapping"]:
                        try:
                            current_mapping = json.loads(row["Mapping"])
                        except:
                            pass
                
                map_data = [{"Value": k, "Rank": v} for k, v in current_mapping.items()]
                map_df = pd.DataFrame(map_data) if map_data else pd.DataFrame(columns=["Value", "Rank"])
                
                edited_map = st.data_editor(
                    map_df,
                    key=f"mapping_{col}",
                    num_rows="dynamic",
                    column_config={
                        "Value": st.column_config.TextColumn("Category Value", required=True),
                        "Rank": st.column_config.NumberColumn("Rank (0 = lowest)", required=True, step=1)
                    }
                )
                
                # Update the mapping in the main dataframe
                new_mapping = {row["Value"]: int(row["Rank"]) for _, row in edited_map.iterrows() if row["Value"]}
                
                # Store updated mapping
                st.session_state[f"encoding_mapping_{col}"] = new_mapping
    
    # Optional encodings section
    st.markdown("**Optional Encodings:**")
    st.caption("Select additional encoding strategies for location and date columns detected in phase 1.")
    
    current_optional_encodings = service.workflow.current_state.get("optional_encodings", {}) if service.workflow else {}
    column_types = service.workflow.current_state.get("column_types", {}) if service.workflow else {}
    
    optional_encodings_ui = {}
    encoded_cols = [row["Column"] for _, row in edited_enc_df.iterrows() if row["Column"]]
    
    # Get location columns from column_types
    location_cols = [col for col in encoded_cols if column_types.get(col) == "location"]
    
    # Get date columns - check if we have access to original dataframe
    # If not available, we'll infer from column_types or encoding type
    date_cols = []
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        date_cols = [col for col in encoded_cols if pd_types.is_datetime64_any_dtype(df[col])]
    else:
        # Fallback: check column_types for datetime
        date_cols = [col for col in encoded_cols if column_types.get(col) == "datetime"]
    
    # Show optional encodings for location columns
    if location_cols:
        with st.expander("Location Column Encodings", expanded=False):
            for col in location_cols:
                current_enc = current_optional_encodings.get(col, {}).get("type", "")
                enc_display = "None"
                if current_enc == "cost_of_living":
                    enc_display = "Cost of Living"
                elif current_enc == "metro_population":
                    enc_display = "Metro Population"
                
                selected = st.selectbox(
                    f"{col}",
                    ["None", "Cost of Living", "Metro Population"],
                    index=["None", "Cost of Living", "Metro Population"].index(enc_display),
                    key=f"opt_enc_location_{col}"
                )
                if selected != "None":
                    if selected == "Cost of Living":
                        optional_encodings_ui[col] = {"type": "cost_of_living", "params": {}}
                    elif selected == "Metro Population":
                        optional_encodings_ui[col] = {"type": "metro_population", "params": {}}
    
    # Show optional encodings for date columns
    if date_cols:
        with st.expander("Date Column Encodings", expanded=False):
            for col in date_cols:
                current_enc = current_optional_encodings.get(col, {}).get("type", "")
                enc_display = "None"
                if current_enc == "normalize_recent":
                    enc_display = "Normalize Recent"
                elif current_enc == "weight_recent":
                    enc_display = "Weight Recent"
                elif current_enc == "least_recent":
                    enc_display = "Least Recent"
                
                selected = st.selectbox(
                    f"{col}",
                    ["None", "Normalize Recent", "Weight Recent", "Least Recent"],
                    index=["None", "Normalize Recent", "Weight Recent", "Least Recent"].index(enc_display),
                    key=f"opt_enc_date_{col}"
                )
                if selected != "None":
                    if selected == "Normalize Recent":
                        optional_encodings_ui[col] = {"type": "normalize_recent", "params": {}}
                    elif selected == "Weight Recent":
                        optional_encodings_ui[col] = {"type": "weight_recent", "params": {}}
                    elif selected == "Least Recent":
                        optional_encodings_ui[col] = {"type": "least_recent", "params": {}}
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Confirm & Continue", type="primary", key="confirm_encoding"):
            # Build modifications
            new_encodings = {}
            for _, row in edited_enc_df.iterrows():
                col_name = row["Column"]
                if not col_name:
                    continue
                    
                enc_type = row["Encoding"]
                
                # Get mapping from session state or parse from text
                mapping = st.session_state.get(f"encoding_mapping_{col_name}", {})
                if not mapping and row["Mapping"]:
                    try:
                        mapping = json.loads(row["Mapping"])
                    except:
                        pass
                
                new_encodings[col_name] = {
                    "type": enc_type,
                    "mapping": mapping,
                    "reasoning": row.get("Notes", "")
                }
            
            modifications = {
                "encodings": new_encodings,
                "optional_encodings": optional_encodings_ui
            }
            
            progress_msg = _get_progress_message(service)
            with st.status(progress_msg, expanded=False) as status:
                status.update(label="Configuring model...", state="running")
                result = service.confirm_encoding(modifications)
                status.update(label="Model configuration complete", state="complete")
            st.session_state["workflow_result"] = result
            st.session_state["workflow_phase"] = "configuration"
            st.rerun()
    
    with col2:
        if st.button("Back to Classification", key="back_to_class"):
            _reset_workflow_state()
            st.rerun()
    
    return None


def _render_configuration_phase(service: WorkflowService, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Render the model configuration review phase. Args: service (WorkflowService): Workflow service. result (Dict[str, Any]): Configuration result. Returns: Optional[Dict[str, Any]]: Config if finalized, None otherwise."""
    st.subheader("Step 3: Model Configuration")
    
    data = result.get("data", {})
    
    # Show reasoning
    reasoning = data.get("reasoning", "")
    if reasoning:
        with st.expander("Agent Reasoning", expanded=True):
            st.markdown(reasoning)
    
    # Features with monotone constraints
    st.markdown("**Feature Constraints:**")
    
    features = data.get("features", [])
    feature_data = []
    for f in features:
        if isinstance(f, dict):
            feature_data.append({
                "Feature": f.get("name", ""),
                "Constraint": f.get("monotone_constraint", 0),
                "Reasoning": f.get("reasoning", "")
            })
        else:
            feature_data.append({"Feature": str(f), "Constraint": 0, "Reasoning": ""})
    
    if not feature_data:
        feature_data = [{"Feature": "", "Constraint": 0, "Reasoning": ""}]
    
    feat_df = pd.DataFrame(feature_data)
    
    edited_feat_df = st.data_editor(
        feat_df,
        key="features_config_editor",
        column_config={
            "Feature": st.column_config.TextColumn("Feature Name"),
            "Constraint": st.column_config.SelectboxColumn(
                "Monotone Constraint",
                options=[-1, 0, 1],
                help="-1: decreasing, 0: none, 1: increasing"
            ),
            "Reasoning": st.column_config.TextColumn("Reasoning")
        },
        hide_index=True,
        width='stretch',
        num_rows="dynamic"
    )
    
    # Quantiles
    st.markdown("**Prediction Quantiles:**")
    quantiles = data.get("quantiles", [0.1, 0.25, 0.5, 0.75, 0.9])
    
    quant_df = pd.DataFrame([{"Quantile": q} for q in quantiles])
    edited_quant_df = st.data_editor(
        quant_df,
        key="quantiles_config_editor",
        num_rows="dynamic",
        column_config={
            "Quantile": st.column_config.NumberColumn(
                "Quantile (0.0-1.0)",
                min_value=0.0,
                max_value=1.0,
                step=0.05
            )
        }
    )
    
    # Location Settings (if location columns exist)
    location_columns = service.workflow.current_state.get("location_columns", []) if service.workflow else []
    if location_columns:
        st.markdown("**Location Proximity Settings:**")
        st.info(f"Detected location columns: {', '.join(location_columns)}")
        
        # Get current location settings from state or default
        current_location_settings = service.workflow.current_state.get("location_settings", {}) if service.workflow else {}
        current_max_dist = current_location_settings.get("max_distance_km", 50)
        current_max_dist = int(current_max_dist) if current_max_dist else 50
        
        max_distance_km = st.slider(
            "Max Distance (km) for Proximity Matching",
            min_value=0,
            max_value=200,
            value=current_max_dist,
            step=5,
            help="Maximum distance in km to consider a candidate 'local' to a target city.",
            key="workflow_max_distance_km"
        )
        
        # Store in session state for later use
        st.session_state["workflow_location_settings"] = {"max_distance_km": max_distance_km}
    else:
        max_distance_km = 50  # Default if no location columns
    
    # Hyperparameters
    st.markdown("**Hyperparameters:**")
    hyperparams = data.get("hyperparameters", {})
    training = hyperparams.get("training", {})
    cv = hyperparams.get("cv", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Training Parameters")
        max_depth = st.number_input("Max Depth", value=training.get("max_depth", 6), min_value=1, max_value=20)
        eta = st.number_input("Learning Rate (eta)", value=training.get("eta", 0.1), min_value=0.01, max_value=1.0, step=0.01)
        subsample = st.number_input("Subsample", value=training.get("subsample", 0.8), min_value=0.1, max_value=1.0, step=0.1)
        colsample = st.number_input("Column Sample", value=training.get("colsample_bytree", 0.8), min_value=0.1, max_value=1.0, step=0.1)
    
    with col2:
        st.write("Cross-Validation Parameters")
        num_boost = st.number_input("Num Boost Rounds", value=cv.get("num_boost_round", 200), min_value=10, max_value=2000)
        nfold = st.number_input("N Folds", value=cv.get("nfold", 5), min_value=2, max_value=10)
        early_stop = st.number_input("Early Stopping", value=cv.get("early_stopping_rounds", 20), min_value=5, max_value=100)
    
    # Action buttons
    action_cols = st.columns([1, 1, 2])
    col1, col2, col3 = action_cols[0], action_cols[1], action_cols[2]
    
    with col1:
        if st.button("Finalize Configuration", type="primary", key="finalize_config"):
            # Build final config from edited values
            final_features = []
            for _, row in edited_feat_df.iterrows():
                if row["Feature"]:
                    final_features.append({
                        "name": row["Feature"],
                        "monotone_constraint": int(row["Constraint"])
                    })
            
            final_quantiles = [float(row["Quantile"]) for _, row in edited_quant_df.iterrows()]
            
            # Get the final config from service and update with user edits
            final_config = service.get_final_config()
            
            if final_config:
                final_config["model"]["features"] = final_features
                final_config["model"]["quantiles"] = final_quantiles
                final_config["model"]["hyperparameters"] = {
                    "training": {
                        "objective": "reg:quantileerror",
                        "tree_method": "hist",
                        "max_depth": int(max_depth),
                        "eta": float(eta),
                        "subsample": float(subsample),
                        "colsample_bytree": float(colsample),
                        "verbosity": 0
                    },
                    "cv": {
                        "num_boost_round": int(num_boost),
                        "nfold": int(nfold),
                        "early_stopping_rounds": int(early_stop),
                        "verbose_eval": False
                    }
                }
                
                # Update location settings if location columns exist
                if location_columns and "workflow_location_settings" in st.session_state:
                    final_config["location_settings"] = st.session_state["workflow_location_settings"]
                    # Also update workflow state
                    if service.workflow:
                        service.workflow.current_state["location_settings"] = st.session_state["workflow_location_settings"]
                
                st.session_state["workflow_result"]["final_config"] = final_config
                st.session_state["workflow_phase"] = "complete"
                st.rerun()
            else:
                st.error("Could not retrieve final configuration")
    
    with col2:
        if st.button("Back to Encoding", key="back_to_encoding"):
            st.session_state["workflow_phase"] = "encoding"
            st.rerun()
    
    return None


def _render_complete_phase(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Render the completion phase with final config. Args: result (Dict[str, Any]): Complete workflow result. Returns: Optional[Dict[str, Any]]: Final config."""
    st.subheader("Configuration Complete!")
    st.success("Your configuration has been generated successfully.")
    
    final_config = result.get("final_config")
    
    if final_config:
        # Show metadata/reasoning
        metadata = final_config.pop("_metadata", {})
        if metadata:
            with st.expander("Generation Summary"):
                if metadata.get("classification_reasoning"):
                    st.markdown(f"**Classification:** {metadata['classification_reasoning']}")
                if metadata.get("encoding_summary"):
                    st.markdown(f"**Encoding:** {metadata['encoding_summary']}")
                if metadata.get("configuration_reasoning"):
                    st.markdown(f"**Configuration:** {metadata['configuration_reasoning']}")
        
        # Show config preview
        with st.expander("Configuration Preview", expanded=True):
            st.json(final_config)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Apply Configuration", type="primary"):
                st.session_state["config_override"] = final_config
                _reset_workflow_state()
                st.rerun()
        
        with col2:
            if st.button("Start Over"):
                _reset_workflow_state()
                st.rerun()
        
        return final_config
    
    return None


def _reset_workflow_state() -> None:
    """Reset all workflow-related session state. Returns: None."""
    keys_to_remove = [
        "workflow_service", "workflow_phase", "workflow_result"
    ]
    for key in list(st.session_state.keys()):
        if key.startswith("encoding_mapping_"):
            keys_to_remove.append(key)
    
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]


# =============================================================================
# Legacy Config Editor Components (kept for manual editing)
# =============================================================================

def render_save_load_controls(current_config_state: Dict[str, Any]) -> None:
    """Renders Save/Load controls."""
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


def render_ranked_mappings_section(config: Dict[str, Any]) -> None:
    """Renders editor for Ranked Mappings (feature_engineering)."""
    st.subheader("Ranked Categories")
    
    if "feature_engineering" not in config:
        config["feature_engineering"] = {
            "ranked_cols": {},
            "proximity_cols": []
        }
        if "mappings" in config and "levels" in config["mappings"]:
            config["feature_engineering"]["ranked_cols"]["Level"] = "levels"
        if "Location" in [f.get("name") for f in config.get("model",{}).get("features",[])]:
             if "proximity_cols" not in config["feature_engineering"]:
                 config["feature_engineering"]["proximity_cols"] = ["Location"]
             elif "Location" not in config["feature_engineering"]["proximity_cols"]:
                 config["feature_engineering"]["proximity_cols"].append("Location")

    fe_config = config["feature_engineering"]
    ranked_cols = fe_config.get("ranked_cols", {})
    mappings = config.get("mappings", {})
    
    st.markdown("##### Mapped Columns")
    
    cols_data = [{"Column": col, "MappingKey": key} for col, key in ranked_cols.items()]
    cols_df = pd.DataFrame(cols_data)
    
    edited_cols = st.data_editor(
        cols_df,
        num_rows="dynamic",
        key="ranked_cols_editor",
        column_config={
            "Column": st.column_config.TextColumn("Column Name", required=True),
            "MappingKey": st.column_config.TextColumn("Mapping Key (in mappings)", required=True)
        }
    )
    
    new_ranked_cols = {}
    for _, row in edited_cols.iterrows():
        if row["Column"] and row["MappingKey"]:
            new_ranked_cols[row["Column"]] = row["MappingKey"]
            if row["MappingKey"] not in mappings:
                mappings[row["MappingKey"]] = {}
                
    config["feature_engineering"]["ranked_cols"] = new_ranked_cols
    
    if new_ranked_cols:
        st.markdown("##### Edit Mappings")
        unique_keys = sorted(list(set(new_ranked_cols.values())))
        selected_key = st.selectbox("Select Mapping to Edit", unique_keys)
        
        if selected_key:
            current_map = mappings.get(selected_key, {})
            data = [{"Category": k, "Rank": v} for k, v in current_map.items()]
            df_map = pd.DataFrame(data)
            
            edited_map_df = st.data_editor(
                df_map,
                num_rows="dynamic",
                key=f"mapping_editor_{selected_key}",
                column_config={
                    "Category": st.column_config.TextColumn("Category Name", required=True),
                    "Rank": st.column_config.NumberColumn("Rank Value", required=True, step=1)
                }
            )
            
            new_map = {}
            for _, row in edited_map_df.iterrows():
                if row["Category"]:
                     new_map[row["Category"]] = int(row["Rank"])
            
            mappings[selected_key] = new_map


def render_location_targets_editor(config: Dict[str, Any]) -> Dict[str, int]:
    """Renders an editor for 'mappings.location_targets'."""
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
    """Renders slider for location settings."""
    st.subheader("Location Settings")
    
    loc_settings = config.get("location_settings", {})
    current_dist = loc_settings.get("max_distance_km", 50)
    # Ensure all values are int type to avoid type mismatch
    current_dist = int(current_dist) if current_dist else 50
    
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
    """Renders editor for 'model' configuration."""
    st.subheader("Model Configuration")
    
    model_config = config.get("model", {})
    
    st.markdown("**Model Variables (Features & Targets)**")
    
    defaults_targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
    current_targets = model_config.get("targets", defaults_targets)
    
    default_features = [
        {"name": "Level_Enc", "monotone_constraint": 1},
        {"name": "Location_Enc", "monotone_constraint": 0},
        {"name": "YearsOfExperience", "monotone_constraint": 1},
        {"name": "YearsAtCompany", "monotone_constraint": 0}
    ]
    current_features = model_config.get("features", default_features)
    
    unified_data = []
    for t in current_targets:
        unified_data.append({"Name": t, "Role": "Target", "Monotone Constraint": 0})
    for f in current_features:
        unified_data.append({
            "Name": f.get("name"),
            "Role": "Feature",
            "Monotone Constraint": f.get("monotone_constraint", 0)
        })
        
    unified_df = pd.DataFrame(unified_data)
    
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
            new_features.append({"name": name, "monotone_constraint": constraint})
    
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
        obj = st.text_input("Objective", value=hp_train.get("objective", "reg:quantileerror"))
        tm = st.selectbox("Tree Method", ["hist", "auto", "exact", "approx"], index=0)
        verb = st.number_input("Verbosity", value=hp_train.get("verbosity", 0))
        
    with cv_col:
        st.write("Cross-Validation")
        hp_cv = current_hyperparams.get("cv", {})
        nbr = st.number_input("Num Boost Rounds", value=hp_cv.get("num_boost_round", 100))
        nfold = st.number_input("N Folds", value=hp_cv.get("nfold", 5))
        esr = st.number_input("Early Stopping Rounds", value=hp_cv.get("early_stopping_rounds", 10))
    
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


# =============================================================================
# Main Entry Point
# =============================================================================

def render_config_ui(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point to render the full config UI.
    
    Includes the agentic workflow wizard and manual editing capabilities.
    """
    st.header("Configuration")
    
    # --- Agentic Config Generator Section ---
    with st.expander("AI-Powered Configuration Wizard", expanded=True):
        st.write("Generate configuration using an intelligent multi-step workflow.")
        
        # Data source selection
        data_source = st.radio(
            "Data Source", 
            ["Use Loaded Training Data", "Upload New CSV"], 
            horizontal=True,
            key="wizard_data_source"
        )
        
        df_to_analyze = None
        
        if data_source == "Use Loaded Training Data":
            if "training_data" in st.session_state:
                df_to_analyze = st.session_state["training_data"]
                st.success(f"Using loaded data ({len(df_to_analyze)} rows).")
            else:
                st.warning("No training data loaded. Please upload a CSV.")
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="wizard_uploader")
            if uploaded_file:
                is_valid, err, df_preview = validate_csv(uploaded_file)
                if is_valid:
                    full_df = load_data_cached(uploaded_file)
                    st.session_state["training_data"] = full_df
                    st.session_state["training_dataset_name"] = uploaded_file.name
                    df_to_analyze = full_df
                    st.success(f"Dataset '{uploaded_file.name}' loaded ({len(full_df)} rows).")
                else:
                    st.error(f"Invalid CSV: {err}")
        
        # Provider selection
        available_providers = get_workflow_providers()
        if not available_providers:
            available_providers = ["openai", "gemini"]
        
        provider = st.selectbox(
            "LLM Provider",
            available_providers,
            index=0,
            key="wizard_provider"
        )
        
        if df_to_analyze is not None:
            st.markdown("---")
            result = render_workflow_wizard(df_to_analyze, provider)
            
            if result:
                st.session_state["config_override"] = result

    st.markdown("---")
    
    # --- Manual Config Editor ---
    st.subheader("Manual Configuration Editor")
    st.write("Edit configuration directly or refine AI-generated settings.")

    if "config_override" in st.session_state:
         config = st.session_state["config_override"]
    
    new_config = copy.deepcopy(config)
    
    if "mappings" not in new_config:
        new_config["mappings"] = {}
        
    render_ranked_mappings_section(new_config)
    new_config["mappings"]["location_targets"] = render_location_targets_editor(new_config)
    new_config["location_settings"] = render_location_settings_editor(new_config)
    new_config["model"] = render_model_config_editor(new_config)
    
    render_save_load_controls(new_config)
    
    return new_config
