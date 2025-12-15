from datetime import datetime
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.utils.geo_utils import GeoMapper
from src.utils.performance import PerformanceMetrics


class RankedCategoryEncoder:
    """Maps ordinal categorical values to integers based on a provided mapping."""

    def __init__(
        self,
        mapping: Optional[Dict[str, int]] = None,
        config: Optional[Dict[str, Any]] = None,
        config_key: Optional[str] = None,
    ) -> None:
        """Initializes encoder. Args: mapping (Optional[Dict[str, int]]): Direct mapping dictionary. config (Optional[Dict[str, Any]]): Configuration dictionary. config_key (Optional[str]): Key in config['mappings'] to load mapping from. Returns: None. Raises: ValueError: If neither mapping nor (config and config_key) are provided."""
        if mapping is not None:
            self.mapping = mapping
        elif config is not None and config_key is not None:
            self.mapping = config.get("mappings", {}).get(config_key, {})
        else:
            if config_key is not None:
                raise ValueError(
                    f"config_key '{config_key}' provided but config is None. "
                    "Please provide config parameter when using config_key."
                )
            self.mapping = {}

    def fit(self, X: Any, y: Optional[Any] = None) -> "RankedCategoryEncoder":
        """Fits the encoder (no-op as mapping is static). Args: X (Any): Input data. y (Optional[Any]): Target data. Returns: RankedCategoryEncoder: self."""
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, list]) -> pd.Series:
        """Transforms categories to their integer representation. Args: X (Union[pd.DataFrame, pd.Series, list]): Input categories. Returns: pd.Series: Integer encoded categories. Unknown categories mapped to -1."""
        with PerformanceMetrics("preprocessing_ranked_encoder_time"):
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]
            return pd.Series(X).map(self.mapping).fillna(-1).astype(int)


class ProximityEncoder:
    """Maps locations to Cost Zones based on proximity to target cities."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ProximityEncoder. Args: config (Optional[Dict[str, Any]]): Configuration dictionary. Returns: None."""
        self.mapper = GeoMapper(config=config)

    def fit(self, X: Any, y: Optional[Any] = None) -> "ProximityEncoder":
        """Fits the encoder (no-op). Args: X (Any): Input data. y (Optional[Any]): Target data. Returns: ProximityEncoder: self."""
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to their cost zone integers. Args: X (Union[pd.DataFrame, pd.Series]): Input locations. Returns: pd.Series: Cost zones (1-4)."""
        with PerformanceMetrics("preprocessing_proximity_encoder_time"):
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]

            def map_loc(loc: Any) -> int:
                if not isinstance(loc, str):
                    return 4
                return self.mapper.get_zone(loc)

            return X.apply(map_loc)


class SampleWeighter:
    """Calculates sample weights based on recency using formula: Weight = 1 / (1 + Age_in_Years)^k."""

    def __init__(
        self,
        k: float = 1.0,
        ref_date: Optional[Union[str, datetime]] = None,
        date_col: str = "Date",
    ) -> None:
        """Initializes the weighter. Args: k (float): Decay parameter. Defaults to 1.0. ref_date (Optional[Union[str, datetime]]): Reference date. Defaults to now. date_col (str): Column name for date if dataframe is passed. Defaults to "Date"."""
        self.k = k
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()
        self.date_col = date_col

    def fit(self, X: Any, y: Optional[Any] = None) -> "SampleWeighter":
        """Fits the weighter (no-op). Args: X (Any): Input data. y (Optional[Any]): Target data. Returns: SampleWeighter: self."""
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculates weights for the input dates. Args: X (Union[pd.DataFrame, pd.Series]): Input dates or dataframe containing date_col. Returns: pd.Series: Calculated weights."""
        if isinstance(X, pd.DataFrame):
            if self.date_col in X.columns:
                X = X[self.date_col]
            else:
                X = X.iloc[:, 0]

        try:
            X_parsed = pd.to_datetime(X, errors="coerce")
            if X_parsed.isna().all():
                return pd.Series(1.0, index=X.index if isinstance(X, pd.Series) else range(len(X)))
            X = X_parsed
        except Exception:
            return pd.Series(1.0, index=X.index if isinstance(X, pd.Series) else range(len(X)))

        age_days = (self.ref_date - X).dt.days
        age_years = age_days / 365.25

        age_years = age_years.clip(lower=0)

        weights = 1 / (1 + age_years) ** self.k
        return weights


class CostOfLivingEncoder:
    """Maps locations to cost of living tiers using the same proximity-based approach as ProximityEncoder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CostOfLivingEncoder. Args: config (Optional[Dict[str, Any]]): Configuration dictionary. Returns: None."""
        self.mapper = GeoMapper(config=config)

    def fit(self, X: Any, y: Optional[Any] = None) -> "CostOfLivingEncoder":
        """Fits the encoder (no-op). Args: X (Any): Input data. y (Optional[Any]): Target data. Returns: CostOfLivingEncoder: self."""
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to their cost of living tier integers. Args: X (Union[pd.DataFrame, pd.Series]): Input locations. Returns: pd.Series: Cost of living tiers (1-4, where 1 is highest cost)."""
        with PerformanceMetrics("preprocessing_cost_of_living_encoder_time"):
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]

            def map_loc(loc: Any) -> int:
                if not isinstance(loc, str):
                    return 4
                return self.mapper.get_zone(loc)

            return X.apply(map_loc)


class MetroPopulationEncoder:
    """Maps locations to metro area population values using a placeholder heuristic based on proximity zones."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MetroPopulationEncoder. Args: config (Optional[Dict[str, Any]]): Configuration dictionary. Returns: None."""
        self.mapper = GeoMapper(config=config)
        self.population_map = {1: 5000000, 2: 2000000, 3: 500000, 4: 100000}

    def fit(self, X: Any, y: Optional[Any] = None) -> "MetroPopulationEncoder":
        """Fits the encoder (no-op). Args: X (Any): Input data. y (Optional[Any]): Target data. Returns: MetroPopulationEncoder: self."""
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to approximate population values. Args: X (Union[pd.DataFrame, pd.Series]): Input locations. Returns: pd.Series: Approximate population values."""
        with PerformanceMetrics("preprocessing_metro_population_encoder_time"):
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]

            def map_loc(loc: Any) -> int:
                if not isinstance(loc, str):
                    return self.population_map[4]
                zone = self.mapper.get_zone(loc)
                return self.population_map.get(zone, self.population_map[4])

            return X.apply(map_loc)


class DateNormalizer:
    """Normalizes dates to 0-1 range based on min/max dates. Supports two modes: normalize_recent (most recent = 1.0) or least_recent (least recent = 0.0)."""

    def __init__(self, mode: str = "normalize_recent") -> None:
        """Initializes the normalizer. Args: mode (str): Normalization mode ('normalize_recent' or 'least_recent')."""
        self.mode = mode
        self.min_date: Optional[pd.Timestamp] = None
        self.max_date: Optional[pd.Timestamp] = None

    def fit(self, X: Any, y: Optional[Any] = None) -> "DateNormalizer":
        """Fits the normalizer by computing min/max dates. Args: X (Any): Input dates. y (Optional[Any]): Target data (ignored). Returns: DateNormalizer: self."""
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        X = pd.to_datetime(X, errors="coerce")
        valid_dates = X.dropna()

        if len(valid_dates) > 0:
            self.min_date = valid_dates.min()
            self.max_date = valid_dates.max()
        else:
            self.min_date = pd.Timestamp.now()
            self.max_date = pd.Timestamp.now()

        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Normalizes dates to 0-1 range. Args: X (Union[pd.DataFrame, pd.Series]): Input dates. Returns: pd.Series: Normalized dates (0.0 to 1.0)."""
        if self.min_date is None or self.max_date is None:
            raise ValueError("DateNormalizer must be fitted before transform")

        with PerformanceMetrics("preprocessing_date_normalizer_time"):
            if isinstance(X, pd.DataFrame):
                X = X.iloc[:, 0]

            X = pd.to_datetime(X, errors="coerce")

            date_range = (self.max_date - self.min_date).total_seconds()

            if date_range == 0:
                return pd.Series([1.0] * len(X), index=X.index)

            if self.mode == "normalize_recent":
                normalized = (X - self.min_date).dt.total_seconds() / date_range
            else:
                normalized = (X - self.min_date).dt.total_seconds() / date_range

            normalized = normalized.fillna(0.0)
            normalized = normalized.clip(0.0, 1.0)

            return normalized


# Backward Compatibility Aliases
LevelEncoder = RankedCategoryEncoder
LocationEncoder = ProximityEncoder
