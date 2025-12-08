import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.config_ui import (
    render_levels_editor, 
    render_location_targets_editor, 
    render_location_settings_editor, 
    render_config_ui,
    render_model_config_editor,
    render_save_load_controls
)
import json

@pytest.fixture
def sample_config():
    return {
        "mappings": {
            "levels": {"E3": 0, "E4": 1},
            "location_targets": {"New York": 1, "Austin": 2}
        },
        "location_settings": {"max_distance_km": 50}
    }

def test_render_levels_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        # Mock data_editor return value (dataframe)
        mock_df = pd.DataFrame([
            {"Level": "E3", "Rank": 0},
            {"Level": "E4", "Rank": 1},
            {"Level": "E5", "Rank": 2} # Added one
        ])
        mock_st.data_editor.return_value = mock_df
        
        updated_levels = render_levels_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Levels Configuration")
        mock_st.data_editor.assert_called_once()
        
        assert updated_levels == {"E3": 0, "E4": 1, "E5": 2}

def test_render_location_targets_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        mock_df = pd.DataFrame([
            {"City": "New York", "Tier/Rank": 1},
            {"City": "Austin", "Tier/Rank": 3} # Changed rank
        ])
        mock_st.data_editor.return_value = mock_df
        
        updated_loc = render_location_targets_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Location Targets")
        assert updated_loc == {"New York": 1, "Austin": 3}

def test_render_location_settings_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        mock_st.slider.return_value = 100
        
        updated_settings = render_location_settings_editor(sample_config)
        
        mock_st.subheader.assert_called_with("Location Settings")
        mock_st.slider.assert_called_with(
            "Max Distance (km) for Proximity Matching",
            min_value=0, max_value=200, value=50, step=5,
            help="Maximum distance in km to consider a candidate 'local' to a target city."
        )
        assert updated_settings == {"max_distance_km": 100}

def test_render_config_ui(sample_config):
    # Integration test of the wrapper
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.render_levels_editor") as mock_levels, \
         patch("src.app.config_ui.render_location_targets_editor") as mock_loc, \
         patch("src.app.config_ui.render_location_settings_editor") as mock_settings, \
         patch("src.app.config_ui.render_model_config_editor") as mock_model, \
         patch("src.app.config_ui.render_save_load_controls"), \
         patch("src.app.config_ui.validate_csv"), \
         patch("src.app.config_ui.ConfigGenerator"):
         
        mock_levels.return_value = {"L": 1}
        mock_loc.return_value = {"C": 2}
        mock_settings.return_value = {"dist": 99}
        mock_model.return_value = {"targets": []}
        
        
        # Mock st.columns (needed for generator section now)
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)] if isinstance(n, int) else [MagicMock() for _ in range(len(n))]
        
        # Ensure file uploader returns None so we don't trigger validation logic in this general test
        mock_st.file_uploader.return_value = None
        
        # Ensure Generate button is not clicked
        mock_st.button.return_value = False

        new_config = render_config_ui(sample_config)
        
        assert new_config["mappings"]["levels"] == {"L": 1}
        assert new_config["mappings"]["location_targets"] == {"C": 2}
        assert new_config["location_settings"] == {"dist": 99}
        assert new_config["model"] == {"targets": []}
        # Verify deep copy didn't affect original if we care (not strictly required by implementation but good practice)
        assert sample_config["location_settings"]["max_distance_km"] == 50

# ... (Previous tests)

def test_render_config_ui_generator_section(sample_config):
    """Test the new Configuration Generator expander logic."""
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.ConfigGenerator") as MockGenerator, \
         patch("src.app.config_ui.validate_csv") as mock_validate, \
         patch("src.app.config_ui.render_levels_editor"), \
         patch("src.app.config_ui.render_location_targets_editor"), \
         patch("src.app.config_ui.render_location_settings_editor"), \
         patch("src.app.config_ui.render_model_config_editor"), \
         patch("src.app.config_ui.render_save_load_controls"):

        # Setup Generator Mock
        mock_gen_instance = MockGenerator.return_value
        mock_gen_instance.generate_config.return_value = {"generated": True}
        
        # Setup Session State
        mock_st.session_state = {}
        
        # Scenario 1: Use Loaded Data (Success)
        training_df = pd.DataFrame({"A": [1]})
        mock_st.radio.return_value = "Use Loaded Training Data"
        mock_st.session_state["training_data"] = training_df
        
        # UI Inputs
        mock_st.checkbox.return_value = True # Use AI
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)] if isinstance(n, int) else [MagicMock() for _ in range(len(n))]
        
        # Selectbox side effect: 1. Provider, 2. Preset
        mock_st.selectbox.side_effect = ["openai", "salary"]
        
        # Button click
        mock_st.button.return_value = True
        
        render_config_ui(sample_config)
        
        # Verify Generator Called - Use direct dataframe comparison
        args, kwargs = mock_gen_instance.generate_config.call_args
        pd.testing.assert_frame_equal(args[0], training_df)
        assert kwargs["use_llm"] is True
        assert kwargs["provider"] == "openai"
        assert kwargs["preset"] == "salary"
        
        # Verify State Updated
        assert mock_st.session_state["config_override"] == {"generated": True}
        mock_st.rerun.assert_called()

def test_render_config_ui_generator_upload_flow(sample_config):
    """Test upload flow in generator."""
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.ConfigGenerator") as MockGenerator, \
         patch("src.app.config_ui.validate_csv") as mock_validate, \
         patch("src.app.config_ui.render_levels_editor"), \
         patch("src.app.config_ui.render_location_targets_editor"), \
         patch("src.app.config_ui.render_location_settings_editor"), \
         patch("src.app.config_ui.render_model_config_editor"), \
         patch("src.app.config_ui.render_save_load_controls"):
         
        mock_st.session_state = {}
        
        # Scenario: Upload New CSV
        mock_st.radio.return_value = "Upload New CSV"
        # Mock uploader returning a truthy value (MagicMock)
        mock_file = MagicMock()
        mock_st.file_uploader.return_value = mock_file
        
        # Validation Success
        upload_df = pd.DataFrame({"B": [2]})
        mock_validate.return_value = (True, None, upload_df)
        
        # Inputs
        mock_st.checkbox.return_value = False # No AI
        mock_st.button.return_value = True
        
        render_config_ui(sample_config)
        
        # Verify Generator Called with heuristics
        # Do not use assert_called_with because DataFrame comparison fails
        assert MockGenerator.return_value.generate_config.called
        args, kwargs = MockGenerator.return_value.generate_config.call_args
        pd.testing.assert_frame_equal(args[0], upload_df)
        assert kwargs["use_llm"] is False
        assert kwargs["provider"] == "openai" # Default
        assert kwargs["preset"] == "none" # Default

def test_render_model_config_editor(sample_config):
    
    with patch("src.app.config_ui.st") as mock_st:
        # Mock st.columns to return 2 objects
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        # Mock returns for data editors
        # 1. Variables Editor (Unified)
        # 2. Quantiles
        mock_st.data_editor.side_effect = [
            pd.DataFrame([
                {"Name": "T1", "Role": "Target", "Monotone Constraint": 0},
                {"Name": "T2", "Role": "Target", "Monotone Constraint": 0},
                {"Name": "F1", "Role": "Feature", "Monotone Constraint": 1},
                {"Name": "Bad", "Role": "Ignore", "Monotone Constraint": 0}
            ]), 
            pd.DataFrame([{"Quantile": 0.1}, {"Quantile": 0.9}])
        ]
        
        # Mock return for inputs
        # 1. Sample Weight (number_input) -> 1.5
        # 2. Verbosity (number_input) -> 1
        # 3. Num Boost Rounds (number_input) -> 200
        # 4. N Folds (number_input) -> 10
        # 5. Early Stopping (number_input) -> 5
        mock_st.number_input.side_effect = [1.5, 1, 200, 10, 5]
        
        # Mock text_input for Objective -> "reg:squaredlogerror"
        mock_st.text_input.return_value = "reg:squaredlogerror"
        
        # Mock selectbox for Tree Method -> "approx"
        mock_st.selectbox.return_value = "approx"
        
        updated_model = render_model_config_editor(sample_config)
        
        # Verify structure
        assert updated_model["targets"] == ["T1", "T2"]
        assert updated_model["quantiles"] == [0.1, 0.9]
        assert updated_model["sample_weight_k"] == 1.5
        
        hp = updated_model["hyperparameters"]
        assert hp["training"]["objective"] == "reg:squaredlogerror"
        assert hp["training"]["tree_method"] == "approx"
        assert hp["training"]["verbosity"] == 1
        
        assert hp["cv"]["num_boost_round"] == 200
        assert hp["cv"]["nfold"] == 10
        
        feat = updated_model["features"]
        assert len(feat) == 1
        assert feat[0]["name"] == "F1"
        assert feat[0]["monotone_constraint"] == 1
        
        # Verify Ignore worked
        assert "Bad" not in updated_model["targets"]
        assert not any(f["name"] == "Bad" for f in feat)

def test_render_save_load_controls_save():
    config = {"a": 1}
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("json.dumps", return_value='{"a": 1}') as mock_json_dumps:
        
        mock_st.file_uploader.return_value = None
        
        render_save_load_controls(config)
        
        mock_st.download_button.assert_called_once()
        args, kwargs = mock_st.download_button.call_args
        assert kwargs["data"] == '{"a": 1}'
        assert kwargs["file_name"] == "config.json"

def test_render_save_load_controls_load_success():
    
    config = {"a": 1}
    loaded_config = {"a": 2}
    
    with patch("src.app.config_ui.st") as mock_st:
        # Mock file uploader returning a file
        mock_file = MagicMock()
        mock_st.file_uploader.return_value = mock_file
        
        # Mock json load
        with patch("json.load", return_value=loaded_config):
            # Mock session state
            mock_st.session_state = {}
            
            render_save_load_controls(config)
            
            assert mock_st.session_state['config_override'] == loaded_config
            mock_st.rerun.assert_called_once()

def test_render_config_ui_uses_override(sample_config):
    
    override_config = {
        "mappings": {"levels": {"O": 99}, "location_targets": {}},
        "location_settings": {"max_distance_km": 10},
        "model": {} 
    }
    
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.render_levels_editor") as mock_levels, \
         patch("src.app.config_ui.render_location_targets_editor") as mock_loc, \
         patch("src.app.config_ui.render_location_settings_editor") as mock_settings, \
         patch("src.app.config_ui.render_model_config_editor") as mock_model, \
         patch("src.app.config_ui.render_save_load_controls") as mock_save_load, \
         patch("src.app.config_ui.validate_csv"), \
         patch("src.app.config_ui.ConfigGenerator"):

        # Use side_effects to return the existing values so config isn't mutated to empty
        mock_levels.side_effect = lambda c: c["mappings"].get("levels", {})
        mock_loc.side_effect = lambda c: c["mappings"].get("location_targets", {})
        mock_settings.side_effect = lambda c: c.get("location_settings", {})
        mock_model.side_effect = lambda c: c.get("model", {})

        # Set overrides in session state
        mock_st.session_state = {"config_override": override_config}
        
        # Mock st.columns
        mock_st.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

        # Ensure file uploader returns None
        mock_st.file_uploader.return_value = None
        
        # Ensure Generate button is not clicked
        mock_st.button.return_value = False

        render_config_ui(sample_config) # pass sample, but expect override used
        
        # Check that render_levels_editor was called with OVERRIDE config, not sample
        # We need to verify the arguments passed to the sub-helpers
        args, _ = mock_levels.call_args
        assert args[0] == override_config




