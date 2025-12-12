"""Configuration generator service providing heuristic-based configuration generation. For AI-powered configuration, use WorkflowService through the Streamlit UI's Configuration Wizard."""

import re
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger


class ConfigGenerator:
    """Service to generate configuration from data using heuristics. For AI-powered configuration, use WorkflowService through the Streamlit UI's Configuration Wizard."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def infer_levels(self, df: pd.DataFrame, level_col: str = "Level") -> Dict[str, int]:
        """Infers level ranking based on heuristics by extracting numeric patterns from level strings. Args: df (pd.DataFrame): Input DataFrame. level_col (str): Name of the level column. Returns: Dict[str, int]: Dictionary mapping level names to rank integers."""
        if level_col not in df.columns:
            return {}

        unique_levels = df[level_col].dropna().unique().tolist()

        def extract_rank(val: str) -> int:
            """Extract numeric rank from level string. Args: val (str): Level value. Returns: int: Extracted rank or -1 if not found."""
            match = re.search(r"\d+", str(val))
            if match:
                return int(match.group())
            return -1

        sorted_levels = sorted(unique_levels, key=lambda x: (extract_rank(x), x))
        return {lvl: i for i, lvl in enumerate(sorted_levels)}

    def infer_locations(self, df: pd.DataFrame, loc_col: str = "Location") -> Dict[str, int]:
        """Infer locations from column. Args: df (pd.DataFrame): Input DataFrame. loc_col (str): Location column name. Returns: Dict[str, int]: Location name to tier mapping."""
        if loc_col not in df.columns:
            return {}

        unique_locs = sorted(df[loc_col].dropna().unique().tolist())
        return {loc: 2 for loc in unique_locs}

    def infer_targets(self, df: pd.DataFrame) -> List[str]:
        """Infer likely target columns based on heuristics. Args: df (pd.DataFrame): Input DataFrame. Returns: List[str]: Likely target column names."""
        target_keywords = [
            "salary",
            "price",
            "cost",
            "total",
            "amount",
            "revenue",
            "profit",
            "value",
            "comp",
            "bonus",
            "stock",
        ]

        targets = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        for col in numeric_cols:
            col_lower = col.lower()
            if any(kw in col_lower for kw in target_keywords):
                targets.append(col)

        return targets

    def infer_features(
        self, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Infers likely feature columns with basic constraints. Args: df (pd.DataFrame): Input DataFrame. exclude_cols (Optional[List[str]]): Columns to exclude. Returns: List[Dict[str, Any]]: List of feature configurations."""
        exclude_cols = exclude_cols or []
        exclude_set = set(exclude_cols)

        id_keywords = ["id", "key", "index", "uuid", "guid"]

        features = []

        for col in df.columns:
            if col in exclude_set:
                continue

            col_lower = col.lower()

            if any(kw in col_lower for kw in id_keywords):
                continue

            constraint = 0
            if "year" in col_lower or "experience" in col_lower or "age" in col_lower:
                constraint = 1
            elif "level" in col_lower or "rank" in col_lower or "grade" in col_lower:
                constraint = 1

            features.append({"name": col, "monotone_constraint": constraint})

        return features

    CONFIG_TEMPLATE = {
        "mappings": {"levels": {}, "location_targets": {}},
        "feature_engineering": {"ranked_cols": {}, "proximity_cols": []},
        "location_settings": {"max_distance_km": 50},
        "model": {
            "targets": [],
            "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
            "sample_weight_k": 1.0,
            "features": [],
            "hyperparameters": {
                "training": {
                    "objective": "reg:quantileerror",
                    "tree_method": "hist",
                    "verbosity": 0,
                },
                "cv": {
                    "num_boost_round": 100,
                    "nfold": 5,
                    "early_stopping_rounds": 10,
                    "verbose_eval": False,
                },
            },
        },
    }

    def generate_config_template(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generates a template config using heuristics. Args: df (pd.DataFrame): Input DataFrame. Returns: Dict[str, Any]: Configuration dictionary."""
        levels = self.infer_levels(df)
        locations = self.infer_locations(df)
        targets = self.infer_targets(df)
        features = self.infer_features(df, exclude_cols=targets)

        config = {
            "mappings": {"levels": levels, "location_targets": locations},
            "feature_engineering": {
                "ranked_cols": {"Level": "levels"} if levels else {},
                "proximity_cols": ["Location"] if locations else [],
            },
            "location_settings": {"max_distance_km": 50},
            "model": {
                "targets": targets,
                "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
                "sample_weight_k": 1.0,
                "features": features,
                "hyperparameters": {
                    "training": {
                        "objective": "reg:quantileerror",
                        "tree_method": "hist",
                        "verbosity": 0,
                    },
                    "cv": {
                        "num_boost_round": 100,
                        "nfold": 5,
                        "early_stopping_rounds": 10,
                        "verbose_eval": False,
                    },
                },
            },
        }

        return config

    def generate_config(
        self,
        df: pd.DataFrame,
        use_llm: bool = False,
        provider: str = "openai",
        preset: str = "none",
    ) -> Dict[str, Any]:
        """Generates configuration from dataframe using heuristics. Note: For AI-powered configuration, use WorkflowService through the Streamlit UI's Configuration Wizard. The use_llm parameter is deprecated. Args: df (pd.DataFrame): Input data. use_llm (bool): Deprecated - use WorkflowService for AI-powered config. provider (str): Deprecated - use WorkflowService for AI-powered config. preset (str): Deprecated - use WorkflowService for AI-powered config. Returns: Dict[str, Any]: Configuration dictionary."""
        if use_llm:
            self.logger.warning(
                "The use_llm parameter is deprecated. For AI-powered configuration, "
                "use the WorkflowService through the Streamlit UI's Configuration Wizard. "
                "Falling back to heuristic-based generation."
            )

        return self.generate_config_template(df)
