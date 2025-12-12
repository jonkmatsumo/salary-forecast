from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger
from src.xgboost.model import SalaryForecaster


class AnalyticsService:
    """Service for data and model analytics."""

    def __init__(self):
        self.logger = get_logger(__name__)

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates high-level statistics for the dataset."""
        if df is None or df.empty:
            return {}

        summary = {"total_samples": len(df), "shape": df.shape}

        # Add unique counts for object/categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            # clean string to be key-friendly
            key_name = f"unique_{col.lower().replace(' ', '_')}"
            summary[key_name] = df[col].nunique()

        return summary

    def get_feature_importance(
        self, model: SalaryForecaster, target: str, quantile_val: float
    ) -> Optional[pd.DataFrame]:
        """Extract feature importance (Gain) for a specific target/quantile model. Args: model (SalaryForecaster): Trained model. target (str): Target column. quantile_val (float): Quantile value. Returns: Optional[pd.DataFrame]: Feature importance DataFrame."""
        model_name = f"{target}_p{int(quantile_val*100)}"

        if not hasattr(model, "models") or model_name not in model.models:
            return None

        xgb_model = model.models[model_name]

        if hasattr(xgb_model, "get_booster"):
            booster = xgb_model.get_booster()
        else:
            booster = xgb_model

        importance = booster.get_score(importance_type="gain")

        if not importance:
            return None

        df_imp = pd.DataFrame(list(importance.items()), columns=["Feature", "Gain"])
        df_imp = df_imp.sort_values(by="Gain", ascending=False)
        return df_imp

    def get_available_targets(self, model: SalaryForecaster) -> List[str]:
        """Returns list of available targets in the model."""
        if hasattr(model, "targets"):
            return model.targets
        # Fallback inspection
        if hasattr(model, "models"):
            keys = list(model.models.keys())
            return sorted(list(set([k.split("_p")[0] for k in keys if "_p" in k])))
        return []

    def get_available_quantiles(
        self, model: SalaryForecaster, target: Optional[str] = None
    ) -> List[float]:
        """Return list of available quantiles. Args: model (SalaryForecaster): Trained model. target (Optional[str]): Target column. Returns: List[float]: Available quantiles."""
        if hasattr(model, "quantiles"):
            return sorted(model.quantiles)

        if hasattr(model, "models"):
            keys = list(model.models.keys())
            if target:
                target_keys = [k for k in keys if k.startswith(f"{target}_p")]
                return sorted([float(k.split("_p")[1]) / 100 for k in target_keys])
            # Try to infer from all keys if no target specified (less accurate if mixed)
            return []
        return []
