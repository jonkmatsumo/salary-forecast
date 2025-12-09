import pandas as pd
from datetime import datetime
from typing import Union, Optional, Any, Dict
from src.utils.config_loader import get_config

class RankedCategoryEncoder:
    """Maps ordinal categorical values to integers based on a provided mapping.

    Attributes:
        mapping (dict): Dictionary mapping category names to integer ranks.
    """
    def __init__(self, mapping: Optional[Dict[str, int]] = None, config_key: Optional[str] = None) -> None:
        """Initialize encoder.
        
        Args:
            mapping: Direct mapping dictionary.
            config_key: Key in config['mappings'] to load mapping from.
        """
        if mapping is not None:
            self.mapping = mapping
        elif config_key is not None:
            config = get_config()
            self.mapping = config.get("mappings", {}).get(config_key, {})
        else:
            self.mapping = {}


    def fit(self, X: Any, y: Optional[Any] = None) -> "RankedCategoryEncoder":
        """Fits the encoder (no-op as mapping is static).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series, list]) -> pd.Series:
        """Transforms categories to their integer representation.
        
        Args:
            X: Input categories.
            
        Returns:
            Integer encoded categories. Unknown categories mapped to -1.
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return pd.Series(X).map(self.mapping).fillna(-1).astype(int)

from src.utils.geo_utils import GeoMapper

class ProximityEncoder:
    """Maps locations to Cost Zones based on proximity to target cities.

    Attributes:
        mapper (GeoMapper): Utility to calculate proximity zones.
    """
    def __init__(self) -> None:
        self.mapper = GeoMapper()


    def fit(self, X: Any, y: Optional[Any] = None) -> "ProximityEncoder":
        """Fits the encoder (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Transforms location names to their cost zone integers.
        
        Args:
            X: Input locations.
            
        Returns:
            Cost zones (1-4).
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        def map_loc(loc: Any) -> int:
            if not isinstance(loc, str):
                return 4
            return self.mapper.get_zone(loc)
            
        return X.apply(map_loc)

class SampleWeighter:
    """Calculates sample weights based on recency.
    
    Formula: Weight = 1 / (1 + Age_in_Years)^k

    Attributes:
        k (float): Decay rate parameter.
        ref_date (datetime): Reference date to calculate age from.
        date_col (str): Name of the date column to use if dataframe passed.
    """
    def __init__(self, k: Optional[float] = None, ref_date: Optional[Union[str, datetime]] = None, date_col: str = "Date") -> None:
        """Initialize the weighter.
        
        Args:
            k: Decay parameter. If None, loads from config.
            ref_date: Reference date. Defaults to now.
            date_col: Column name for date if dataframe is passed. Defaults to "Date".
        """
        if k is None:
            config = get_config()
            self.k = config["model"].get("sample_weight_k", 1.0)
        else:
            self.k = k
            
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()
        self.date_col = date_col

    def fit(self, X: Any, y: Optional[Any] = None) -> "SampleWeighter":
        """Fits the weighter (no-op).
        
        Args:
            X: Input data.
            y: Target data (optional).
            
        Returns:
            self
        """
        return self

    def transform(self, X: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        """Calculates weights for the input dates.
        
        Args:
            X: Input dates or dataframe containing date_col.
            
        Returns:
            Calculated weights.
        """
        if isinstance(X, pd.DataFrame):
            if self.date_col in X.columns:
                X = X[self.date_col]
            else:
                X = X.iloc[:, 0]
        
        X = pd.to_datetime(X)
        
        age_days = (self.ref_date - X).dt.days
        age_years = age_days / 365.25
        
        age_years = age_years.clip(lower=0)
        
        weights = 1 / (1 + age_years) ** self.k
        return weights

# Backward Compatibility Aliases
LevelEncoder = RankedCategoryEncoder
LocationEncoder = ProximityEncoder
