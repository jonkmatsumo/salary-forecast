import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from src.app.config_ui import render_levels_editor, render_location_targets_editor, render_location_settings_editor, render_config_ui

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
            min_value=0, max_value=200, value=50, step=5
        )
        assert updated_settings == {"max_distance_km": 100}

def test_render_config_ui(sample_config):
    # Integration test of the wrapper
    with patch("src.app.config_ui.st") as mock_st, \
         patch("src.app.config_ui.render_levels_editor") as mock_levels, \
         patch("src.app.config_ui.render_location_targets_editor") as mock_loc, \
         patch("src.app.config_ui.render_location_settings_editor") as mock_settings:
        
        mock_levels.return_value = {"L": 1}
        mock_loc.return_value = {"C": 2}
        mock_settings.return_value = {"dist": 99}
        
        # Mock st.columns inside render_model_config_editor which is called by render_config_ui
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        new_config = render_config_ui(sample_config)
        
        assert new_config["mappings"]["levels"] == {"L": 1}
        assert new_config["mappings"]["location_targets"] == {"C": 2}
        assert new_config["location_settings"] == {"dist": 99}
        # Verify deep copy didn't affect original if we care (not strictly required by implementation but good practice)
        assert sample_config["location_settings"]["max_distance_km"] == 50

def test_render_model_config_editor(sample_config):
    from src.app.config_ui import render_model_config_editor
    
    with patch("src.app.config_ui.st") as mock_st:
        # Mock st.columns to return 2 objects
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        
        # Mock returns for data editors
        # 1. Targets
        mock_st.data_editor.side_effect = [
            pd.DataFrame([{"Target": "T1"}, {"Target": "T2"}]), # Targets
            pd.DataFrame([{"Quantile": 0.1}, {"Quantile": 0.9}]), # Quantiles
            pd.DataFrame([{"name": "F1", "monotone_constraint": 1}]) # Features
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
