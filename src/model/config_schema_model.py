from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field, model_validator


class Mappings(BaseModel):
    levels: Dict[str, int] = Field(
        default_factory=dict, description="Map of level name to rank (0-based)"
    )
    location_targets: Dict[str, int] = Field(
        default_factory=dict, description="Map of location name to tier (1=High)"
    )


class FeatureConfig(BaseModel):
    name: str
    monotone_constraint: Literal[-1, 0, 1] = Field(
        description="Monotonic constraint: 1 (increasing), 0 (none), -1 (decreasing)"
    )


class ModelConfig(BaseModel):
    targets: List[str] = Field(description="List of target columns to predict", min_length=1)
    features: List[FeatureConfig] = Field(
        default_factory=list, description="List of feature configurations"
    )
    quantiles: List[float] = Field(
        default=[0.1, 0.25, 0.5, 0.75, 0.9], description="Quantiles to predict", min_length=1
    )
    sample_weight_k: float = Field(default=1.0, ge=0.0)
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Hyperparameters for training and CV"
    )

    @model_validator(mode="after")
    def validate_quantiles_range(self) -> "ModelConfig":
        """Validate that quantiles are between 0 and 1. Returns: ModelConfig: Self for chaining."""
        for q in self.quantiles:
            if not 0 <= q <= 1:
                raise ValueError(f"Quantiles must be between 0 and 1, got {q}")
        return self

    @model_validator(mode="after")
    def validate_features_unique(self) -> "ModelConfig":
        """Validate that feature names are unique. Returns: ModelConfig: Self for chaining."""
        names = [f.name for f in self.features]
        if len(names) != len(set(names)):
            raise ValueError("Feature names must be unique")
        return self


class Config(BaseModel):
    mappings: Mappings = Field(default_factory=Mappings)
    location_settings: Dict[str, float] = Field(default_factory=lambda: {"max_distance_km": 50.0})
    model: ModelConfig
    optional_encodings: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional encoding strategies for columns (e.g., cost_of_living for locations, normalize_recent for dates)",
    )


def validate_config_dict(config: Dict[str, Any]) -> Config:
    """Validate a configuration dictionary using Pydantic. Args: config (Dict[str, Any]): Configuration dictionary. Returns: Config: Validated Config object. Raises: ValidationError: If config is invalid."""
    from typing import cast

    return cast(Config, Config.model_validate(config))
