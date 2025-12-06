import pandas as pd
import numpy as np
from datetime import datetime

class LevelEncoder:
    """
    Maps company levels to ordinal integers.
    E3 -> 0, E4 -> 1, E5 -> 2, E6 -> 3, E7 -> 4
    """
    def __init__(self):
        self.mapping = {
            "E3": 0,
            "E4": 1,
            "E5": 2,
            "E6": 3,
            "E7": 4
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a Series or list of level strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return X.map(self.mapping).fillna(-1).astype(int)

class LocationEncoder:
    """
    Maps locations to Cost Zones (1-4).
    Zone 1: NYC, SF Bay Area
    Zone 2: LA, Seattle, DC
    Zone 3: Austin, Boston, Denver, Houston, Portland, Sacramento, San Diego
    Zone 4: Rest of World / National Average
    """
    def __init__(self):
        self.zone_mapping = {
            # Zone 1
            "New York": 1, "New York City": 1, "NYC": 1, "Newark": 1, "Jersey City": 1,
            "San Francisco": 1, "SF": 1, "Oakland": 1, "San Jose": 1, "Palo Alto": 1, "Bay Area": 1, "Mountain View": 1, "Sunnyvale": 1,
            
            # Zone 2
            "Los Angeles": 2, "LA": 2,
            "Seattle": 2, "Bellevue": 2, "Redmond": 2,
            "Washington DC": 2, "DC": 2, "Washington": 2, "Arlington": 2,
            
            # Zone 3
            "Austin": 3,
            "Boston": 3, "Cambridge": 3,
            "Denver": 3, "Boulder": 3,
            "Houston": 3,
            "Portland": 3,
            "Sacramento": 3,
            "San Diego": 3,
        }
        # Zone 4 is default

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        # Helper to map a single value
        def map_loc(loc):
            if not isinstance(loc, str):
                return 4
            for key, zone in self.zone_mapping.items():
                if key.lower() in loc.lower():
                    return zone
            return 4 # Default
            
        return X.apply(map_loc)

class SampleWeighter:
    """
    Calculates sample weights based on recency.
    Weight = 1 / (1 + Age_in_Years)^k
    """
    def __init__(self, k=1.0, ref_date=None):
        self.k = k
        self.ref_date = pd.to_datetime(ref_date) if ref_date else datetime.now()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a Series of dates
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        
        X = pd.to_datetime(X)
        
        # Calculate age in years
        age_days = (self.ref_date - X).dt.days
        age_years = age_days / 365.25
        
        # Clip negative age (future dates) to 0
        age_years = age_years.clip(lower=0)
        
        weights = 1 / (1 + age_years) ** self.k
        return weights
