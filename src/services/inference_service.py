"""Inference service for model predictions and validation."""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.services.model_registry import ModelRegistry
from src.utils.logger import get_logger
from src.xgboost.model import SalaryForecaster


class ModelNotFoundError(Exception):
    """Raised when a model cannot be found by run_id."""

    pass


class InvalidInputError(Exception):
    """Raised when input features are invalid."""

    pass


class ModelSchema:
    """Represents the schema of a trained model."""

    def __init__(self, model: SalaryForecaster) -> None:
        """Initialize model schema from a trained model.

        Args:
            model (SalaryForecaster): Trained model instance.
        """
        self.ranked_features: List[str] = list(model.ranked_encoders.keys())
        self.proximity_features: List[str] = list(model.proximity_encoders.keys())
        self.numerical_features: List[str] = [
            f
            for f in model.feature_names
            if f not in self.ranked_features
            and f not in self.proximity_features
            and f not in [f"{h}_Enc" for h in self.ranked_features]
            and f not in [f"{h}_Enc" for h in self.proximity_features]
        ]
        self.all_feature_names: List[str] = model.feature_names
        self.targets: List[str] = model.targets
        self.quantiles: List[float] = model.quantiles


class ValidationResult:
    """Result of input feature validation."""

    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None) -> None:
        """Initialize validation result.

        Args:
            is_valid (bool): Whether validation passed.
            errors (Optional[List[str]]): List of error messages if invalid.
        """
        self.is_valid = is_valid
        self.errors = errors or []


class PredictionResult:
    """Result of a prediction request."""

    def __init__(
        self,
        predictions: Dict[str, Dict[str, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize prediction result.

        Args:
            predictions (Dict[str, Dict[str, float]]): Predictions by target and quantile.
            metadata (Optional[Dict[str, Any]]): Optional metadata.
        """
        self.predictions = predictions
        self.metadata = metadata or {}


class InferenceService:
    """Service for model inference operations."""

    def __init__(self, model_registry: Optional[ModelRegistry] = None) -> None:
        """Initialize inference service.

        Args:
            model_registry (Optional[ModelRegistry]): Model registry instance. If None, creates a new one.
        """
        self.logger = get_logger(__name__)
        self.registry = model_registry or ModelRegistry()
        self._model_cache: Dict[str, SalaryForecaster] = {}

    def load_model(self, run_id: str) -> SalaryForecaster:
        """Load a model from the registry.

        Args:
            run_id (str): MLflow run ID.

        Returns:
            SalaryForecaster: Loaded model.

        Raises:
            ModelNotFoundError: If model cannot be loaded.
        """
        if run_id in self._model_cache:
            self.logger.debug(f"Returning cached model for run_id: {run_id}")
            return self._model_cache[run_id]

        try:
            self.logger.info(f"Loading model from registry: {run_id}")
            model = self.registry.load_model(run_id)
            self._model_cache[run_id] = model
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model {run_id}: {e}", exc_info=True)
            raise ModelNotFoundError(f"Model with run_id '{run_id}' not found: {str(e)}") from e

    def get_model_schema(self, model: SalaryForecaster) -> ModelSchema:
        """Get the schema of a model.

        Args:
            model (SalaryForecaster): Model instance.

        Returns:
            ModelSchema: Model schema.
        """
        return ModelSchema(model)

    def validate_input_features(
        self, model: SalaryForecaster, features: Dict[str, Any]
    ) -> ValidationResult:
        """Validate input features against model schema.

        Args:
            model (SalaryForecaster): Model instance.
            features (Dict[str, Any]): Input feature dictionary.

        Returns:
            ValidationResult: Validation result.
        """
        schema = self.get_model_schema(model)
        errors: List[str] = []

        required_ranked = set(schema.ranked_features)
        required_proximity = set(schema.proximity_features)
        required_numerical = set(schema.numerical_features)

        provided_features = set(features.keys())

        missing_ranked = required_ranked - provided_features
        if missing_ranked:
            errors.append(f"Missing ranked features: {', '.join(sorted(missing_ranked))}")

        missing_proximity = required_proximity - provided_features
        if missing_proximity:
            errors.append(f"Missing proximity features: {', '.join(sorted(missing_proximity))}")

        missing_numerical = required_numerical - provided_features
        if missing_numerical:
            errors.append(f"Missing numerical features: {', '.join(sorted(missing_numerical))}")

        for col in schema.ranked_features:
            if col in features:
                val = features[col]
                encoder = model.ranked_encoders[col]
                valid_levels = list(encoder.mapping.keys())
                if val not in valid_levels:
                    errors.append(
                        f"Invalid value for ranked feature '{col}': '{val}'. "
                        f"Valid values: {', '.join(valid_levels[:10])}{'...' if len(valid_levels) > 10 else ''}"
                    )

        for col in schema.numerical_features:
            if col in features:
                val = features[col]
                if not isinstance(val, (int, float)):
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        errors.append(
                            f"Invalid value for numerical feature '{col}': '{val}'. Expected numeric value."
                        )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def predict(self, model: SalaryForecaster, features: Dict[str, Any]) -> PredictionResult:
        """Execute prediction for given features.

        Args:
            model (SalaryForecaster): Model instance.
            features (Dict[str, Any]): Input feature dictionary.

        Returns:
            PredictionResult: Prediction result.

        Raises:
            InvalidInputError: If input features are invalid.
        """
        validation_result = self.validate_input_features(model, features)
        if not validation_result.is_valid:
            raise InvalidInputError(
                f"Invalid input features: {'; '.join(validation_result.errors)}"
            )

        try:
            input_df = pd.DataFrame([features])
            raw_predictions = model.predict(input_df)

            formatted_predictions: Dict[str, Dict[str, float]] = {}
            for target, preds in raw_predictions.items():
                formatted_predictions[target] = {}
                for q_key, val_array in preds.items():
                    formatted_predictions[target][q_key] = float(val_array[0])

            metadata = {
                "model_targets": model.targets,
                "model_quantiles": model.quantiles,
            }

            if hasattr(model, "proximity_encoders") and "Location" in model.proximity_encoders:
                location_val = features.get("Location")
                if location_val:
                    encoder = model.proximity_encoders["Location"]
                    if hasattr(encoder, "mapper") and hasattr(encoder.mapper, "get_zone"):
                        try:
                            zone = encoder.mapper.get_zone(location_val)
                            metadata["location_zone"] = zone
                        except Exception:
                            pass

            return PredictionResult(predictions=formatted_predictions, metadata=metadata)
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            raise InvalidInputError(f"Prediction failed: {str(e)}") from e

    def format_predictions(self, predictions: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Format predictions for display.

        Args:
            predictions (Dict[str, Dict[str, float]]): Raw predictions.

        Returns:
            List[Dict[str, Any]]: Formatted predictions as list of rows.
        """
        formatted: List[Dict[str, Any]] = []
        for target, preds in predictions.items():
            row: Dict[str, Any] = {"Component": target}
            row.update(preds)
            formatted.append(row)
        return formatted
