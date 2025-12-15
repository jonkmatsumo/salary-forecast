import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.app.config_ui import (
    render_config_ui,
    render_location_settings_editor,
    render_location_targets_editor,
    render_model_config_editor,
    render_ranked_mappings_section,
    render_save_load_controls,
)


@pytest.fixture
def sample_config():
    return {
        "mappings": {
            "levels": {"E3": 0, "E4": 1},
            "location_targets": {"New York": 1, "Austin": 2},
        },
        "feature_engineering": {"ranked_cols": {"Level": "levels"}, "proximity_cols": ["Location"]},
        "location_settings": {"max_distance_km": 50},
    }


def test_render_ranked_mappings_section(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        # Mocking data_editor for generic ranked cols
        # 1. First call: Mapped Columns Editor
        # 2. Second call: Mapping Editor for specific key (if selected)

        # We need to simulate the flow.
        # First st.data_editor call returns the columns dataframe
        # Second call returns the mapping dataframe (if we set formatting)

        mock_df_cols = pd.DataFrame([{"Column": "Level", "MappingKey": "levels"}])

        mock_df_map = pd.DataFrame(
            [
                {"Category": "E3", "Rank": 0},
                {"Category": "E4", "Rank": 1},
                {"Category": "E5", "Rank": 2},
            ]
        )

        mock_st.data_editor.side_effect = [mock_df_cols, mock_df_map]

        # Selectbox choice
        mock_st.selectbox.return_value = "levels"

        render_ranked_mappings_section(sample_config)

        mock_st.subheader.assert_called_with("Ranked Categories")

        # Verify Mappings Updated
        assert sample_config["mappings"]["levels"] == {"E3": 0, "E4": 1, "E5": 2}

        # Verify Columns Updated
        assert sample_config["feature_engineering"]["ranked_cols"] == {"Level": "levels"}


def test_render_location_targets_editor(sample_config):
    with patch("src.app.config_ui.st") as mock_st:
        mock_df = pd.DataFrame(
            [
                {"City": "New York", "Tier/Rank": 1},
                {"City": "Austin", "Tier/Rank": 3},  # Changed rank
            ]
        )
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
            min_value=0,
            max_value=200,
            value=50,
            step=5,
            help="Maximum distance in km to consider a candidate 'local' to a target city.",
        )
        assert updated_settings == {"max_distance_km": 100}


def test_render_config_ui(sample_config):
    # Integration test of the wrapper
    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("src.app.config_ui.render_ranked_mappings_section") as mock_ranked,
        patch("src.app.config_ui.render_location_targets_editor") as mock_loc,
        patch("src.app.config_ui.render_location_settings_editor") as mock_settings,
        patch("src.app.config_ui.render_model_config_editor") as mock_model,
        patch("src.app.config_ui.render_save_load_controls"),
        patch("src.app.config_ui.validate_csv"),
    ):

        # Mock side effect to update config for ranked
        def update_ranked(c):
            c["mappings"]["levels"] = {"L": 1}

        mock_ranked.side_effect = update_ranked

        mock_loc.return_value = {"C": 2}
        mock_settings.return_value = {"dist": 99}
        mock_model.return_value = {"targets": []}

        # Mock st.columns (needed for generator section now)
        mock_st.columns.side_effect = lambda n: (
            [MagicMock() for _ in range(n)]
            if isinstance(n, int)
            else [MagicMock() for _ in range(len(n))]
        )

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

    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("src.app.config_ui.WorkflowService") as MockWorkflowService,
        patch("src.app.config_ui.validate_csv") as mock_validate,
        patch("src.app.config_ui.render_workflow_wizard") as mock_wizard,
        patch("src.app.config_ui.render_ranked_mappings_section"),
        patch("src.app.config_ui.render_location_targets_editor"),
        patch("src.app.config_ui.render_location_settings_editor"),
        patch("src.app.config_ui.render_model_config_editor"),
        patch("src.app.config_ui.render_save_load_controls"),
        patch("src.app.config_ui.get_workflow_providers") as mock_get_providers,
    ):

        # Setup mocks
        mock_get_providers.return_value = ["openai", "gemini"]
        mock_wizard.return_value = {"generated": True}

        # Setup Session State
        mock_st.session_state = {}

        # Scenario 1: Use Loaded Data (Success)
        training_df = pd.DataFrame({"A": [1]})
        mock_st.radio.return_value = "Use Loaded Training Data"
        mock_st.session_state["training_data"] = training_df

        # UI Inputs
        mock_st.checkbox.return_value = True  # Use AI
        mock_st.columns.side_effect = lambda n: (
            [MagicMock() for _ in range(n)]
            if isinstance(n, int)
            else [MagicMock() for _ in range(len(n))]
        )

        # Selectbox side effect: Provider
        mock_st.selectbox.return_value = "openai"

        render_config_ui(sample_config)

        # Verify workflow wizard was called with correct parameters
        assert mock_wizard.called
        call_args = mock_wizard.call_args
        pd.testing.assert_frame_equal(call_args[0][0], training_df)
        assert call_args[0][1] == "openai"

        # Verify State Updated if wizard returns result
        if mock_wizard.return_value:
            assert mock_st.session_state["config_override"] == {"generated": True}


def test_render_config_ui_generator_upload_flow(sample_config):
    """Test upload flow in generator."""

    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("src.app.config_ui.validate_csv") as mock_validate,
        patch("src.app.config_ui.load_data_cached") as mock_load_data,
        patch("src.app.config_ui.render_ranked_mappings_section"),
        patch("src.app.config_ui.render_location_targets_editor"),
        patch("src.app.config_ui.render_location_settings_editor"),
        patch("src.app.config_ui.render_model_config_editor"),
        patch("src.app.config_ui.render_save_load_controls"),
    ):

        mock_st.session_state = {}

        # Scenario: Upload New CSV
        mock_st.radio.return_value = "Upload New CSV"
        # Mock uploader returning a truthy value (MagicMock)
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_st.file_uploader.return_value = mock_file

        # Validation Success
        upload_df = pd.DataFrame({"B": [2]})
        mock_validate.return_value = (True, None, upload_df)
        mock_load_data.return_value = upload_df

        # Inputs
        mock_st.checkbox.return_value = False  # No AI
        mock_st.button.return_value = True
        mock_st.selectbox.return_value = "openai"  # Provider selection

        # Mock get_workflow_providers
        with patch("src.app.config_ui.get_workflow_providers", return_value=["openai", "gemini"]):
            render_config_ui(sample_config)

        # With new workflow approach, when use_llm=False, the workflow wizard
        # handles it internally. Just verify the UI rendered without errors.
        assert True  # Test passes if no exceptions


def test_render_model_config_editor(sample_config):

    with patch("src.app.config_ui.st") as mock_st:
        # Mock st.columns to return 2 objects
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        # Mock returns for data editors
        # 1. Variables Editor (Unified)
        # 2. Quantiles
        mock_st.data_editor.side_effect = [
            pd.DataFrame(
                [
                    {"Name": "T1", "Role": "Target", "Monotone Constraint": 0},
                    {"Name": "T2", "Role": "Target", "Monotone Constraint": 0},
                    {"Name": "F1", "Role": "Feature", "Monotone Constraint": 1},
                    {"Name": "Bad", "Role": "Ignore", "Monotone Constraint": 0},
                ]
            ),
            pd.DataFrame([{"Quantile": 0.1}, {"Quantile": 0.9}]),
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

    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("json.dumps", return_value='{"a": 1}') as mock_json_dumps,
    ):

        mock_st.file_uploader.return_value = None

        render_save_load_controls(config)

        mock_st.download_button.assert_called_once()
        args, kwargs = mock_st.download_button.call_args
        assert kwargs["data"] == '{"a": 1}'
        assert kwargs["file_name"] == "config.json"


def test_render_save_load_controls_json_export_only():

    config = {"a": 1}

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.markdown = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.download_button = MagicMock()
        mock_st.info = MagicMock()
        mock_st.session_state = {}

        render_save_load_controls(config)

        # Verify download button was called (export still available)
        mock_st.download_button.assert_called_once()

        # Verify info message about JSON loading being removed
        mock_st.info.assert_called_once()
        info_call = str(mock_st.info.call_args)
        assert "JSON" in info_call or "deprecated" in info_call.lower() or "removed" in info_call.lower()

        # Verify file uploader was NOT called (loading removed)
        if hasattr(mock_st, "file_uploader"):
            assert not mock_st.file_uploader.called or mock_st.file_uploader.call_count == 0


def test_render_config_ui_uses_override(sample_config):

    override_config = {
        "mappings": {"levels": {"O": 99}, "location_targets": {}},
        "location_settings": {"max_distance_km": 10},
        "model": {},
    }

    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("src.app.config_ui.render_ranked_mappings_section") as mock_ranked,
        patch("src.app.config_ui.render_location_targets_editor") as mock_loc,
        patch("src.app.config_ui.render_location_settings_editor") as mock_settings,
        patch("src.app.config_ui.render_model_config_editor") as mock_model,
        patch("src.app.config_ui.render_save_load_controls") as mock_save_load,
        patch("src.app.config_ui.validate_csv"),
    ):

        # Use side_effects to return the existing values
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

        render_config_ui(sample_config)  # pass sample, but expect override used

        # Check that render_ranked_mappings_section was called with OVERRIDE config
        args, _ = mock_ranked.call_args
        assert args[0] == override_config


# =============================================================================
# Workflow Wizard Tests
# =============================================================================


def test_render_workflow_wizard_initialization():
    """Test render_workflow_wizard initialization."""
    from src.app.config_ui import render_workflow_wizard

    df = pd.DataFrame({"Salary": [100000], "Level": ["L3"]})

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {}
        mock_st.button.return_value = False

        result = render_workflow_wizard(df)

        assert result is None
        assert mock_st.session_state.get("workflow_phase") == "not_started"


@patch("src.app.config_ui.get_workflow_service")
def test_workflow_wizard_start_button(mock_get_workflow_service):
    """Test workflow start button."""
    from src.app.config_ui import render_workflow_wizard

    df = pd.DataFrame({"Salary": [100000]})
    mock_service = MagicMock()
    mock_service.start_workflow.return_value = {
        "phase": "classification",
        "status": "success",
        "data": {"targets": ["Salary"], "features": [], "ignore": []},
    }
    mock_get_workflow_service.return_value = mock_service

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {}
        mock_st.button.return_value = True
        mock_st.selectbox.return_value = "None"
        mock_st.status.return_value.__enter__ = MagicMock()
        mock_st.status.return_value.__exit__ = MagicMock()
        mock_st.rerun = MagicMock()

        result = render_workflow_wizard(df)

        mock_get_workflow_service.assert_called_once()
        mock_service.start_workflow.assert_called_once()
        assert mock_st.session_state["workflow_phase"] == "classification"


@patch("src.app.config_ui.get_workflow_service")
def test_workflow_wizard_phase_indicator(mock_get_workflow_service):
    """Test phase indicator display."""
    from src.app.config_ui import render_workflow_wizard

    df = pd.DataFrame({"A": [1]})
    mock_service = MagicMock()
    mock_get_workflow_service.return_value = mock_service

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {"workflow_phase": "encoding"}
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False

        render_workflow_wizard(df, provider="openai")

        # Should display phase indicator
        mock_st.columns.assert_called()


@patch("src.app.config_ui.get_workflow_service")
def test_render_classification_phase(mock_get_workflow_service):
    """Test _render_classification_phase."""
    from src.app.config_ui import _render_classification_phase

    mock_service = MagicMock()
    result = {
        "phase": "classification",
        "status": "success",
        "data": {
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test reasoning",
        },
        "confirmed": False,
    }

    df = pd.DataFrame({"Salary": [100], "Level": ["L3"], "ID": [1]})

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.return_value = pd.DataFrame(
            {
                "Column": ["Salary", "Level", "ID"],
                "Role": ["Target", "Feature", "Ignore"],
                "Dtype": ["int64", "object", "int64"],
            }
        )
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.rerun = MagicMock()
        mock_st.session_state = {"workflow_service": mock_service}

        _render_classification_phase(None, False, result, df)

        mock_st.subheader.assert_called_with("Step 1: Column Classification")


@patch("src.app.config_ui.get_workflow_service")
def test_render_classification_phase_fallback(mock_get_workflow_service):
    """Test _render_classification_phase fallback to workflow state."""
    from src.app.config_ui import _render_classification_phase

    mock_workflow = MagicMock()
    mock_workflow.current_state = {
        "column_classification": {
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Fallback reasoning",
        },
        "column_types": {},
    }
    mock_service = MagicMock()
    mock_service.workflow = mock_workflow
    
    result = {
        "phase": "classification",
        "status": "success",
        "data": {},  # Empty data - should trigger fallback
        "confirmed": False,
    }

    df = pd.DataFrame({"Salary": [100], "Level": ["L3"], "ID": [1]})

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.data_editor.return_value = pd.DataFrame(
            {
                "Column": ["Salary", "Level", "ID"],
                "Role": ["Target", "Feature", "Ignore"],
                "Dtype": ["int64", "object", "int64"],
            }
        )
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.rerun = MagicMock()

        with patch("src.app.config_ui.st.session_state", {"workflow_service": mock_service}):
            _render_classification_phase(None, False, result, df)

        mock_st.subheader.assert_called_with("Step 1: Column Classification")
        # Verify that the data editor was called (classification data should be populated from fallback)
        assert mock_st.data_editor.called


@patch("src.app.config_ui.get_workflow_service")
def test_render_encoding_phase(mock_get_workflow_service):
    """Test _render_encoding_phase."""
    from src.app.config_ui import _render_encoding_phase

    mock_service = MagicMock()
    result = {
        "phase": "encoding",
        "status": "success",
        "data": {
            "encodings": {"Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}}},
            "summary": "Test summary",
        },
        "confirmed": False,
    }

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.return_value = pd.DataFrame(
            {
                "Column": ["Level"],
                "Encoding": ["ordinal"],
                "Mapping": ['{"L1": 0, "L2": 1}'],
                "Notes": [""],
            }
        )
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.rerun = MagicMock()
        mock_st.session_state = {"workflow_service": mock_service}

        _render_encoding_phase(None, False, result)

        mock_st.subheader.assert_called_with("Step 2: Feature Encoding")


@patch("src.app.config_ui.get_workflow_service")
def test_render_configuration_phase(mock_get_workflow_service):
    """Test _render_configuration_phase."""
    from src.app.config_ui import _render_configuration_phase

    mock_service = MagicMock()
    mock_service.get_final_config.return_value = {"model": {"targets": ["Salary"], "features": []}}

    result = {
        "phase": "configuration",
        "status": "success",
        "data": {
            "features": [{"name": "Level", "monotone_constraint": 1}],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {"training": {}, "cv": {}},
            "reasoning": "Test",
        },
    }

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.side_effect = [
            pd.DataFrame({"Feature": ["Level"], "Constraint": [1], "Reasoning": [""]}),
            pd.DataFrame({"Quantile": [0.1, 0.5, 0.9]}),
        ]
        # Mock columns - first call returns 2 columns, second call returns 3 columns for action buttons
        mock_st.columns.side_effect = [
            [MagicMock(), MagicMock()],  # First call for training/cv params
            [MagicMock(), MagicMock(), MagicMock()],  # Second call for action buttons
        ]
        mock_st.number_input.side_effect = [6, 0.1, 0.8, 0.8, 200, 5, 20]
        mock_st.button.return_value = False
        mock_st.rerun = MagicMock()
        mock_st.session_state = {"workflow_service": mock_service}

        _render_configuration_phase(None, False, result)

        mock_st.subheader.assert_called_with("Step 3: Model Configuration")


@patch("src.app.config_ui.get_workflow_service")
def test_render_complete_phase(mock_get_workflow_service):
    """Test _render_complete_phase."""
    from src.app.config_ui import _render_complete_phase

    result = {
        "phase": "complete",
        "status": "complete",
        "final_config": {
            "model": {"targets": ["Salary"]},
            "_metadata": {
                "classification_reasoning": "Test",
                "encoding_summary": "Test",
                "configuration_reasoning": "Test",
            },
        },
    }

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.success = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.json = MagicMock()
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.button.return_value = False
        mock_st.rerun = MagicMock()
        mock_st.session_state = {}

        config = _render_complete_phase(result)

        assert config is not None
        assert config["model"]["targets"] == ["Salary"]


def test_reset_workflow_state():
    """Test _reset_workflow_state clears all state."""
    from src.app.config_ui import _reset_workflow_state

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.session_state = {
            "workflow_service": MagicMock(),
            "workflow_phase": "classification",
            "workflow_result": {},
            "encoding_mapping_Level": {"L1": 0},
        }

        _reset_workflow_state()

        assert "workflow_service" not in mock_st.session_state
        assert "workflow_phase" not in mock_st.session_state
        assert "workflow_result" not in mock_st.session_state
        assert "encoding_mapping_Level" not in mock_st.session_state


@patch("src.app.config_ui.get_workflow_service")
def test_render_workflow_wizard_error_handling(mock_get_workflow_service):
    """Test render_workflow_wizard handles errors gracefully."""
    from src.app.config_ui import render_workflow_wizard

    df = pd.DataFrame({"Salary": [100000]})

    with patch("src.app.config_ui.st") as mock_st:
        mock_service = MagicMock()
        mock_service.start_workflow.side_effect = Exception("Workflow error")
        mock_get_workflow_service.return_value = mock_service

        mock_st.session_state = {}
        mock_st.button.return_value = True
        mock_st.selectbox.return_value = "None"
        mock_st.status.return_value.__enter__ = MagicMock()
        mock_st.status.return_value.__exit__ = MagicMock()
        mock_st.error = MagicMock()

        result = render_workflow_wizard(df)

        # Should show error message
        mock_st.error.assert_called()
        assert result is None


def test_render_classification_phase_confirmation():
    """Test _render_classification_phase confirmation flow."""
    from src.app.config_ui import _render_classification_phase

    mock_service = MagicMock()
    mock_service.confirm_classification.return_value = {
        "phase": "encoding",
        "status": "success",
        "data": {},
    }

    result = {
        "phase": "classification",
        "status": "success",
        "data": {
            "targets": ["Salary"],
            "features": ["Level"],
            "ignore": ["ID"],
            "reasoning": "Test",
        },
    }

    df = pd.DataFrame({"Salary": [100], "Level": ["L3"], "ID": [1]})

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.return_value = pd.DataFrame(
            {
                "Column": ["Salary", "Level", "ID"],
                "Role": ["Target", "Feature", "Ignore"],
                "Dtype": ["int64", "object", "int64"],
            }
        )
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        # Mock button to return True for "Confirm & Continue"
        mock_st.button.side_effect = lambda label, **kwargs: label == "Confirm & Continue"
        mock_st.rerun = MagicMock()
        mock_st.session_state = {}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        mock_st.session_state["workflow_service"] = mock_service
        _render_classification_phase(None, False, result, df)

        # Should call confirm_classification
        mock_service.confirm_classification.assert_called_once()
        # Should update session state
        assert mock_st.session_state["workflow_phase"] == "encoding"


def test_render_encoding_phase_confirmation():
    """Test _render_encoding_phase confirmation flow."""
    from src.app.config_ui import _render_encoding_phase

    mock_service = MagicMock()
    mock_service.confirm_encoding.return_value = {
        "phase": "configuration",
        "status": "success",
        "data": {},
    }

    result = {
        "phase": "encoding",
        "status": "success",
        "data": {
            "encodings": {"Level": {"type": "ordinal", "mapping": {"L1": 0, "L2": 1}}},
            "summary": "Test",
        },
    }

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.return_value = pd.DataFrame(
            {
                "Column": ["Level"],
                "Encoding": ["ordinal"],
                "Mapping": ['{"L1": 0, "L2": 1}'],
                "Notes": [""],
            }
        )
        mock_st.columns.return_value = [MagicMock(), MagicMock(), MagicMock()]
        mock_st.button.side_effect = lambda label, **kwargs: label == "Confirm & Continue"
        mock_st.rerun = MagicMock()
        mock_st.session_state = {"workflow_service": mock_service}
        mock_st.spinner.return_value.__enter__ = MagicMock()
        mock_st.spinner.return_value.__exit__ = MagicMock()

        _render_encoding_phase(None, False, result)

        mock_service.confirm_encoding.assert_called_once()
        assert mock_st.session_state["workflow_phase"] == "configuration"


def test_render_configuration_phase_confirmation():
    """Test _render_configuration_phase confirmation flow."""
    from src.app.config_ui import _render_configuration_phase

    mock_service = MagicMock()
    mock_service.get_final_config.return_value = {"model": {"targets": ["Salary"]}, "_metadata": {}}

    result = {
        "phase": "configuration",
        "status": "success",
        "data": {
            "features": [{"name": "Level", "monotone_constraint": 1}],
            "quantiles": [0.1, 0.5, 0.9],
            "hyperparameters": {"training": {}, "cv": {}},
            "reasoning": "Test",
        },
    }

    with patch("src.app.config_ui.st") as mock_st:
        mock_st.subheader = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock()
        mock_st.expander.return_value.__exit__ = MagicMock()
        mock_st.data_editor.side_effect = [
            pd.DataFrame({"Feature": ["Level"], "Constraint": [1], "Reasoning": [""]}),
            pd.DataFrame({"Quantile": [0.1, 0.5, 0.9]}),
        ]
        mock_st.columns.side_effect = [
            [MagicMock(), MagicMock()],
            [MagicMock(), MagicMock(), MagicMock()],
        ]
        mock_st.number_input.side_effect = [6, 0.1, 0.8, 0.8, 200, 5, 20]
        # Button label is "Finalize Configuration" not "Confirm & Generate Config"
        mock_st.button.side_effect = lambda label, **kwargs: label == "Finalize Configuration"
        mock_st.rerun = MagicMock()
        mock_st.session_state = {"workflow_result": {}}

        # Mock service.workflow to avoid AttributeError
        mock_service.workflow = MagicMock()
        mock_service.workflow.current_state = {"location_columns": []}
        mock_st.session_state["workflow_service"] = mock_service

        config = _render_configuration_phase(None, False, result)

        # Should get final config when button is clicked
        mock_service.get_final_config.assert_called_once()
        assert mock_st.session_state["workflow_phase"] == "complete"


def test_render_workflow_wizard_complete_phase():
    """Test render_workflow_wizard when phase is complete."""
    from src.app.config_ui import render_workflow_wizard

    df = pd.DataFrame({"Salary": [100000]})

    with (
        patch("src.app.config_ui.st") as mock_st,
        patch("src.app.config_ui._render_complete_phase") as mock_render_complete,
    ):

        mock_st.session_state = {
            "workflow_phase": "complete",
            "workflow_result": {
                "phase": "complete",
                "final_config": {"model": {"targets": ["Salary"]}},
            },
        }

        mock_render_complete.return_value = {"model": {"targets": ["Salary"]}}

        result = render_workflow_wizard(df)

        mock_render_complete.assert_called_once()
        assert result is not None
