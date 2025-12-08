import pandas as pd
import re
from typing import Dict, Any, List, Optional
from src.utils.logger import get_logger

class ConfigGenerator:
    """Service to generate configuration from data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)

    def infer_levels(self, df: pd.DataFrame, level_col: str = "Level") -> Dict[str, int]:
        """Infers level ranking based on heuristics.
        
        Logic:
        1. Extract first integer found in string (e.g. "E5" -> 5, "L3" -> 3).
        2. Sort by integer, then by string.
        3. If no integer, sort alphabetically.
        """
        if level_col not in df.columns:
            return {}
            
        unique_levels = df[level_col].dropna().unique().tolist()
        
        def extract_rank(val: str):
            # Find first integer
            match = re.search(r'\d+', str(val))
            if match:
                return int(match.group())
            return -1 # Default low rank for non-numeric levels
            
        # Sort tuple: (extracted_rank, string_val)
        # We assume higher number = higher rank usually
        sorted_levels = sorted(unique_levels, key=lambda x: (extract_rank(x), x))
        
        # Map to 0-indexed rank
        return {lvl: i for i, lvl in enumerate(sorted_levels)}

    def infer_locations(self, df: pd.DataFrame, loc_col: str = "Location") -> Dict[str, int]:
        """Infers locations from column. Default tier is 2."""
        if loc_col not in df.columns:
            return {}
            
        unique_locs = sorted(df[loc_col].dropna().unique().tolist())
        return {loc: 2 for loc in unique_locs} # Default tier 2

    def generate_config_template(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates a template config based on the dataframe structure."""
        
        # Defaults
        levels = self.infer_levels(df)
        locations = self.infer_locations(df)
        
        return {
            "mappings": {
                "levels": levels,
                "location_targets": locations
            },
            "location_settings": {
                "max_distance_km": 50
            },
            "model": {
                "targets": [], # To be filled by user
                "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
                "sample_weight_k": 1.0,
                "features": [], # To be filled by user
                "hyperparameters": {
                    "training": {"objective": "reg:quantileerror", "tree_method": "hist", "verbosity": 0},
                    "cv": {"num_boost_round": 100, "nfold": 5, "early_stopping_rounds": 10}
                }
            }
        }
