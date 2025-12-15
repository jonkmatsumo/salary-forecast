"""Unit tests for InferenceService."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from conftest import create_test_config  # noqa: E402

from src.services.inference_service import (  # noqa: E402
    InferenceService,
    InvalidInputError,
    ModelNotFoundError,
)


class TestInferenceService(unittest.TestCase):
    """Tests for InferenceService."""

    def setUp(self):
        """Set up test fixtures."""
        self.service = InferenceService()
        self.test_config = create_test_config()

    def test_load_model_success(self):
        """Test loading a model successfully."""
        mock_model = MagicMock()
        mock_registry = MagicMock()
        mock_registry.load_model.return_value = mock_model

        service = InferenceService(model_registry=mock_registry)
        result = service.load_model("test_run_id")

        self.assertEqual(result, mock_model)
        mock_registry.load_model.assert_called_once_with("test_run_id")

    def test_load_model_cached(self):
        """Test that models are cached after first load."""
        mock_model = MagicMock()
        mock_registry = MagicMock()
        mock_registry.load_model.return_value = mock_model

        service = InferenceService(model_registry=mock_registry)
        result1 = service.load_model("test_run_id")
        result2 = service.load_model("test_run_id")

        self.assertEqual(result1, mock_model)
        self.assertEqual(result2, mock_model)
        mock_registry.load_model.assert_called_once_with("test_run_id")

    def test_load_model_not_found(self):
        """Test loading a non-existent model raises ModelNotFoundError."""
        mock_registry = MagicMock()
        mock_registry.load_model.side_effect = Exception("Model not found")

        service = InferenceService(model_registry=mock_registry)

        with self.assertRaises(ModelNotFoundError):
            service.load_model("nonexistent_run_id")

    def test_get_model_schema(self):
        """Test getting model schema."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock()}
        mock_model.proximity_encoders = {"Location": MagicMock()}
        mock_model.feature_names = ["Level_Enc", "Location_Enc", "YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.1, 0.5, 0.9]

        schema = self.service.get_model_schema(mock_model)

        self.assertEqual(schema.ranked_features, ["Level"])
        self.assertEqual(schema.proximity_features, ["Location"])
        self.assertEqual(schema.numerical_features, ["YearsOfExperience"])
        self.assertEqual(schema.targets, ["BaseSalary"])
        self.assertEqual(schema.quantiles, [0.1, 0.5, 0.9])

    def test_validate_input_features_valid(self):
        """Test validation with valid input features."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock(mapping={"L3": 0, "L4": 1, "L5": 2})}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {"Level": "L4", "YearsOfExperience": 5}

        result = self.service.validate_input_features(mock_model, features)

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_validate_input_features_missing_ranked(self):
        """Test validation fails when ranked features are missing."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock()}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["Level_Enc"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {}

        result = self.service.validate_input_features(mock_model, features)

        self.assertFalse(result.is_valid)
        self.assertIn("Missing ranked features", result.errors[0])

    def test_validate_input_features_missing_proximity(self):
        """Test validation fails when proximity features are missing."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {}
        mock_model.proximity_encoders = {"Location": MagicMock()}
        mock_model.feature_names = ["Location_Enc"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {}

        result = self.service.validate_input_features(mock_model, features)

        self.assertFalse(result.is_valid)
        self.assertIn("Missing proximity features", result.errors[0])

    def test_validate_input_features_missing_numerical(self):
        """Test validation fails when numerical features are missing."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {}

        result = self.service.validate_input_features(mock_model, features)

        self.assertFalse(result.is_valid)
        self.assertIn("Missing numerical features", result.errors[0])

    def test_validate_input_features_invalid_ranked_value(self):
        """Test validation fails when ranked feature value is invalid."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock(mapping={"L3": 0, "L4": 1, "L5": 2})}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["Level_Enc"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {"Level": "L99"}

        result = self.service.validate_input_features(mock_model, features)

        self.assertFalse(result.is_valid)
        self.assertIn("Invalid value for ranked feature", result.errors[0])

    def test_validate_input_features_invalid_numerical_value(self):
        """Test validation fails when numerical feature value is invalid."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {"YearsOfExperience": "not_a_number"}

        result = self.service.validate_input_features(mock_model, features)

        self.assertFalse(result.is_valid)
        self.assertIn("Invalid value for numerical feature", result.errors[0])

    def test_predict_success(self):
        """Test successful prediction."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock(mapping={"L4": 1})}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["Level_Enc", "YearsOfExperience"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]
        mock_model.predict.return_value = {"BaseSalary": {"p50": pd.Series([150000.0])}}

        features = {"Level": "L4", "YearsOfExperience": 5}

        result = self.service.predict(mock_model, features)

        self.assertIsNotNone(result)
        self.assertIn("BaseSalary", result.predictions)
        self.assertEqual(result.predictions["BaseSalary"]["p50"], 150000.0)

    def test_predict_invalid_input(self):
        """Test prediction fails with invalid input."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {"Level": MagicMock()}
        mock_model.proximity_encoders = {}
        mock_model.feature_names = ["Level_Enc"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]

        features = {}

        with self.assertRaises(InvalidInputError):
            self.service.predict(mock_model, features)

    def test_predict_with_location_metadata(self):
        """Test prediction includes location zone metadata when available."""
        mock_model = MagicMock()
        mock_model.ranked_encoders = {}
        mock_mapper = MagicMock()
        mock_mapper.get_zone.return_value = "Zone1"
        mock_encoder = MagicMock()
        mock_encoder.mapper = mock_mapper
        mock_model.proximity_encoders = {"Location": mock_encoder}
        mock_model.feature_names = ["Location_Enc"]
        mock_model.targets = ["BaseSalary"]
        mock_model.quantiles = [0.5]
        mock_model.predict.return_value = {"BaseSalary": {"p50": pd.Series([150000.0])}}

        features = {"Location": "San Francisco, CA"}

        result = self.service.predict(mock_model, features)

        self.assertIn("location_zone", result.metadata)
        self.assertEqual(result.metadata["location_zone"], "Zone1")

    def test_format_predictions(self):
        """Test formatting predictions for display."""
        predictions = {
            "BaseSalary": {"p10": 120000.0, "p50": 150000.0, "p90": 180000.0},
            "TotalComp": {"p10": 140000.0, "p50": 170000.0, "p90": 200000.0},
        }

        formatted = self.service.format_predictions(predictions)

        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0]["Component"], "BaseSalary")
        self.assertEqual(formatted[0]["p10"], 120000.0)
        self.assertEqual(formatted[0]["p50"], 150000.0)
        self.assertEqual(formatted[0]["p90"], 180000.0)
