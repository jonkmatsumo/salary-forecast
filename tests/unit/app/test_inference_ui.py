import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd

from src.app.inference_ui import render_inference_ui, render_model_information
from src.services.inference_service import ModelSchema


class TestRenderModelInformation(unittest.TestCase):
    """Tests for render_model_information function."""

    @patch("src.app.inference_ui.st")
    def test_render_model_information_with_run(self, mock_st):
        """Verify model information displays correctly when run data is available."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {"Level": MagicMock(mapping={"E3": 0, "E4": 1, "E5": 2})}
        mock_forecaster.proximity_encoders = {"Location": MagicMock()}
        mock_forecaster.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]

        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = ["Location"]
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]

        run_id = "test_run_12345"
        runs = [
            {
                "run_id": run_id,
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
                "tags.additional_tag": "test_tag",
            }
        ]

        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)

        render_model_information(mock_forecaster, mock_schema, run_id, runs)

        mock_st.subheader.assert_called_with("Model Information")

        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        self.assertTrue(any("Run ID" in call for call in markdown_calls))

    @patch("src.app.inference_ui.st")
    def test_render_model_information_no_run(self, mock_st):
        """Verify graceful handling when run metadata is missing."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {}
        mock_forecaster.proximity_encoders = {}
        mock_forecaster.feature_names = []

        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = []
        mock_schema.proximity_features = []
        mock_schema.numerical_features = []
        mock_schema.all_feature_names = []

        run_id = "missing_run"
        runs = [{"run_id": "other_run", "start_time": datetime(2023, 1, 1)}]

        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)

        render_model_information(mock_forecaster, mock_schema, run_id, runs)

        mock_st.info.assert_called_with("Metadata not available")

    @patch("src.app.inference_ui.st")
    def test_render_model_information_feature_info(self, mock_st):
        """Verify feature information is displayed for model transparency."""
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {
            "Level": MagicMock(mapping={"E3": 0, "E4": 1, "E5": 2, "E6": 3, "E7": 4, "E8": 5})
        }
        mock_forecaster.proximity_encoders = {"Location": MagicMock()}
        mock_forecaster.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]

        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = ["Location"]
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]

        run_id = "test_run"
        runs = [{"run_id": run_id, "start_time": datetime(2023, 1, 1)}]

        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=None)

        render_model_information(mock_forecaster, mock_schema, run_id, runs)

        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list]
        self.assertTrue(any("Ranked Features" in call for call in markdown_calls))
        self.assertTrue(any("Proximity Features" in call for call in markdown_calls))
        self.assertTrue(any("Total Features" in call for call in markdown_calls))


class TestRenderInferenceUI(unittest.TestCase):
    """Tests for render_inference_ui function."""

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.ModelRegistry")
    def test_render_inference_ui_no_models(self, mock_registry_class, mock_get_api_client, mock_st):
        """Verify user is informed when no models are available."""
        mock_get_api_client.return_value = None  # API disabled
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = []

        render_inference_ui()

        mock_st.warning.assert_called_with(
            "No trained models found in MLflow. Please train a new model."
        )
        mock_st.selectbox.assert_not_called()

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.ModelRegistry")
    @patch("src.app.inference_ui.render_model_information")
    def test_render_inference_ui_model_loading_error(
        self,
        mock_render_info,
        mock_registry_class,
        mock_get_inference_service,
        mock_get_api_client,
        mock_st,
    ):
        """Verify graceful error handling when model loading fails."""
        mock_get_api_client.return_value = None  # API disabled
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )

        mock_st.session_state = {}

        mock_inference_service = mock_get_inference_service.return_value
        from src.services.inference_service import ModelNotFoundError

        mock_inference_service.load_model.side_effect = ModelNotFoundError("Model not found")

        render_inference_ui()

        # Should show error message
        mock_st.error.assert_called()
        error_call = mock_st.error.call_args[0][0]
        self.assertIn("Failed to load model", error_call)

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.get_analytics_service")
    @patch("src.app.inference_ui.ModelRegistry")
    @patch("src.app.inference_ui.render_model_information")
    def test_render_inference_ui_success(
        self,
        mock_render_info,
        mock_registry_class,
        mock_get_analytics_service,
        mock_get_inference_service,
        mock_get_api_client,
        mock_st,
    ):
        """Test render_inference_ui with successful model loading."""
        mock_get_api_client.return_value = None  # API disabled
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        # Mock forecaster
        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {"Level": MagicMock(mapping={"E3": 0, "E4": 1})}
        mock_forecaster.proximity_encoders = {}
        mock_forecaster.feature_names = ["Level_Enc", "YearsOfExperience"]

        # Mock schema
        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = []
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_schema.targets = ["BaseSalary"]
        mock_schema.quantiles = [0.5]

        mock_inference_service = mock_get_inference_service.return_value
        mock_inference_service.load_model.return_value = mock_forecaster
        mock_inference_service.get_model_schema.return_value = mock_schema

        # Mock selectbox
        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )

        # Mock session state
        mock_st.session_state = {}

        # Mock form and columns
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        # Mock form_submit_button to return False (form not submitted)
        # Note: form_submit_button is called on st, not on the form context
        mock_st.form_submit_button.return_value = False
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]

        # Mock analytics service
        mock_analytics = mock_get_analytics_service.return_value
        mock_analytics.get_feature_importance.return_value = pd.DataFrame()

        # Mock expander for Model Analysis
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        render_inference_ui()

        # Verify model information was rendered
        mock_render_info.assert_called_once()
        # Verify schema was passed
        call_args = mock_render_info.call_args
        self.assertEqual(call_args[0][1], mock_schema)  # schema is second positional arg

        # Verify header was set
        mock_st.header.assert_called_with("Salary Inference")

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.get_analytics_service")
    @patch("src.app.inference_ui.ModelRegistry")
    @patch("src.app.inference_ui.render_model_information")
    def test_render_inference_ui_uses_inference_service(
        self,
        mock_render_info,
        mock_registry_class,
        mock_get_analytics_service,
        mock_get_inference_service,
        mock_get_api_client,
        mock_st,
    ):
        """Verify that InferenceService is used for model loading and schema retrieval."""
        mock_get_api_client.return_value = None  # API disabled
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        mock_forecaster = MagicMock()
        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = []
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_schema.targets = ["BaseSalary"]
        mock_schema.quantiles = [0.5]

        mock_inference_service = mock_get_inference_service.return_value
        mock_inference_service.load_model.return_value = mock_forecaster
        mock_inference_service.get_model_schema.return_value = mock_schema

        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )
        mock_st.session_state = {}
        mock_st.form_submit_button.return_value = False
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_analytics = mock_get_analytics_service.return_value
        mock_analytics.get_feature_importance.return_value = pd.DataFrame()
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        render_inference_ui()

        # Verify InferenceService methods were called
        mock_get_inference_service.assert_called()
        mock_inference_service.load_model.assert_called_once_with("test_run")
        mock_inference_service.get_model_schema.assert_called_once_with(mock_forecaster)

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.get_analytics_service")
    @patch("src.app.inference_ui.ModelRegistry")
    def test_render_inference_ui_prediction_success(
        self,
        mock_registry_class,
        mock_get_analytics_service,
        mock_get_inference_service,
        mock_get_api_client,
        mock_st,
    ):
        """Verify successful prediction flow using InferenceService."""
        from src.services.inference_service import PredictionResult

        mock_get_api_client.return_value = None
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        mock_forecaster = MagicMock()
        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = []
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_schema.targets = ["BaseSalary"]
        mock_schema.quantiles = [0.5]

        mock_inference_service = mock_get_inference_service.return_value
        mock_inference_service.load_model.return_value = mock_forecaster
        mock_inference_service.get_model_schema.return_value = mock_schema

        # Mock prediction result
        prediction_result = PredictionResult(
            predictions={"BaseSalary": {"p50": 150000.0}}, metadata={"location_zone": None}
        )
        mock_inference_service.predict.return_value = prediction_result

        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )
        mock_st.session_state = {}

        # Mock form submission
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.form_submit_button.return_value = True  # Form submitted
        mock_st.columns.return_value = [MagicMock(), MagicMock()]

        # selectbox is called multiple times - need to handle all calls
        selectbox_calls = [
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run",  # Model selection
            "E4",  # Level selection (in form)
            "BaseSalary",  # Target selection (in model analysis)
            "P50",  # Quantile selection (in model analysis)
        ]
        call_count = [0]

        def selectbox_side_effect(*args, **kwargs):
            result = selectbox_calls[call_count[0] % len(selectbox_calls)]
            call_count[0] += 1
            return result

        mock_st.selectbox.side_effect = selectbox_side_effect

        mock_st.number_input.return_value = 5  # YearsOfExperience
        mock_st.text_input.return_value = "New York"  # Location (if needed)
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander
        mock_analytics = mock_get_analytics_service.return_value
        mock_analytics.get_feature_importance.return_value = pd.DataFrame()

        render_inference_ui()

        # Verify prediction was called with InferenceService
        mock_inference_service.predict.assert_called_once()
        call_args = mock_inference_service.predict.call_args
        self.assertEqual(call_args[0][0], mock_forecaster)  # First arg is model
        self.assertIn("Level", call_args[0][1])  # Second arg is features dict

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.ModelRegistry")
    def test_render_inference_ui_prediction_invalid_input_error(
        self, mock_registry_class, mock_get_inference_service, mock_get_api_client, mock_st
    ):
        """Verify InvalidInputError is handled gracefully during prediction."""
        from src.services.inference_service import InvalidInputError

        mock_get_api_client.return_value = None
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        mock_forecaster = MagicMock()
        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = []
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_schema.targets = ["BaseSalary"]
        mock_schema.quantiles = [0.5]

        mock_inference_service = mock_get_inference_service.return_value
        mock_inference_service.load_model.return_value = mock_forecaster
        mock_inference_service.get_model_schema.return_value = mock_schema
        mock_inference_service.predict.side_effect = InvalidInputError(
            "Invalid input features: Missing ranked features: Level"
        )

        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )
        mock_st.session_state = {}
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.form_submit_button.return_value = True
        mock_st.columns.return_value = [MagicMock(), MagicMock()]
        mock_st.number_input.return_value = 5
        mock_st.expander.return_value = MagicMock()

        render_inference_ui()

        # Verify error was displayed
        mock_st.error.assert_called()
        error_call = mock_st.error.call_args[0][0]
        self.assertIn("Invalid input", error_call)

    @patch("src.app.inference_ui.st")
    @patch("src.app.inference_ui.get_api_client")
    @patch("src.app.inference_ui.get_inference_service")
    @patch("src.app.inference_ui.get_analytics_service")
    @patch("src.app.inference_ui.ModelRegistry")
    def test_render_inference_ui_uses_schema_for_features(
        self,
        mock_registry_class,
        mock_get_analytics_service,
        mock_get_inference_service,
        mock_get_api_client,
        mock_st,
    ):
        """Verify that schema is used for building feature input forms."""
        mock_get_api_client.return_value = None
        mock_registry = mock_registry_class.return_value
        mock_registry.list_models.return_value = [
            {
                "run_id": "test_run",
                "start_time": datetime(2023, 1, 1, 12, 0),
                "tags.model_type": "XGBoost",
                "metrics.cv_mean_score": 0.95,
                "tags.dataset_name": "Test Dataset",
            }
        ]

        mock_forecaster = MagicMock()
        mock_forecaster.ranked_encoders = {"Level": MagicMock(mapping={"E3": 0, "E4": 1})}
        mock_schema = MagicMock(spec=ModelSchema)
        mock_schema.ranked_features = ["Level"]
        mock_schema.proximity_features = ["Location"]
        mock_schema.numerical_features = ["YearsOfExperience"]
        mock_schema.all_feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        mock_schema.targets = ["BaseSalary"]
        mock_schema.quantiles = [0.5]

        mock_inference_service = mock_get_inference_service.return_value
        mock_inference_service.load_model.return_value = mock_forecaster
        mock_inference_service.get_model_schema.return_value = mock_schema

        mock_st.selectbox.return_value = (
            "2023-01-01 12:00 | XGBoost | Test Dataset | CV:0.9500 | ID:test_run"
        )
        mock_st.session_state = {}
        mock_form = MagicMock()
        mock_st.form.return_value.__enter__ = MagicMock(return_value=mock_form)
        mock_st.form.return_value.__exit__ = MagicMock(return_value=None)
        mock_st.form_submit_button.return_value = False
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2]
        mock_analytics = mock_get_analytics_service.return_value
        mock_analytics.get_feature_importance.return_value = pd.DataFrame()
        mock_expander = MagicMock()
        mock_expander.__enter__ = MagicMock(return_value=MagicMock())
        mock_expander.__exit__ = MagicMock(return_value=None)
        mock_st.expander.return_value = mock_expander

        render_inference_ui()

        # Verify selectbox was called for ranked features (iterating over schema.ranked_features)
        selectbox_calls = [call[0][0] for call in mock_st.selectbox.call_args_list]
        # Should have at least one call for Level (ranked feature)
        self.assertTrue(any("Level" in str(call) for call in selectbox_calls if call))

        # Verify text_input was called for proximity features
        text_input_calls = [call[0][0] for call in mock_st.text_input.call_args_list]
        self.assertTrue(any("Location" in str(call) for call in text_input_calls if call))

        # Verify number_input was called for numerical features
        number_input_calls = [call[0][0] for call in mock_st.number_input.call_args_list]
        self.assertTrue(
            any("YearsOfExperience" in str(call) for call in number_input_calls if call)
        )
