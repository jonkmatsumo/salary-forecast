from typing import Any, Callable, Dict, List, Optional, Tuple

import optuna
import pandas as pd

import xgboost as xgb
from src.utils.config_loader import get_config
from src.utils.logger import get_logger
from src.xgboost.preprocessing import (
    CostOfLivingEncoder,
    DateNormalizer,
    MetroPopulationEncoder,
    ProximityEncoder,
    RankedCategoryEncoder,
    SampleWeighter,
)


class QuantileForecaster:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.logger = get_logger(__name__)
        self.models: Dict[str, Any] = {}

        if config is None:
            config = get_config()

        self.config = config
        model_config = config["model"]
        self.model_config: Dict[str, Any] = model_config

        self.targets: List[str] = model_config["targets"]
        self.quantiles: List[float] = model_config["quantiles"]

        self.ranked_encoders: Dict[str, RankedCategoryEncoder] = {}
        self.proximity_encoders: Dict[str, ProximityEncoder] = {}
        self.optional_encoders: Dict[str, Any] = {}

        fe_config = config.get("feature_engineering", {})

        if not fe_config:
            if "mappings" in config and "levels" in config["mappings"]:
                fe_config = {"ranked_cols": {"Level": "levels"}, "proximity_cols": ["Location"]}

        for col, map_key in fe_config.get("ranked_cols", {}).items():
            self.ranked_encoders[col] = RankedCategoryEncoder(config_key=map_key)

        for col in fe_config.get("proximity_cols", []):
            self.proximity_encoders[col] = ProximityEncoder()

        optional_encodings = config.get("optional_encodings", {})
        self.date_weight_col: Optional[str] = None

        for col, enc_config in optional_encodings.items():
            enc_type = enc_config.get("type", "")
            if enc_type == "cost_of_living":
                self.optional_encoders[col] = CostOfLivingEncoder()
            elif enc_type == "metro_population":
                self.optional_encoders[col] = MetroPopulationEncoder()
            elif enc_type == "normalize_recent":
                self.optional_encoders[col] = DateNormalizer(mode="normalize_recent")
            elif enc_type == "least_recent":
                self.optional_encoders[col] = DateNormalizer(mode="least_recent")
            elif enc_type == "weight_recent":
                self.date_weight_col = col

        k = model_config.get("sample_weight_k", 1.0)
        date_col = model_config.get("date_col", "Date")
        weight_date_col = self.date_weight_col if self.date_weight_col else date_col
        self.weighter = SampleWeighter(k=k, date_col=weight_date_col)

        self.features_config: List[Dict[str, Any]] = model_config["features"]
        self.feature_names: List[str] = [f["name"] for f in self.features_config]

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses input data. Args: X (pd.DataFrame): Input data. Returns: pd.DataFrame: Preprocessed data."""
        X_proc = X.copy()

        for col, encoder in self.ranked_encoders.items():
            if col in X_proc.columns:
                X_proc[f"{col}_Enc"] = encoder.transform(X_proc[col])

        for col, encoder in self.proximity_encoders.items():
            if col in X_proc.columns:
                X_proc[f"{col}_Enc"] = encoder.transform(X_proc[col])

        optional_feature_names = []
        for col, encoder in self.optional_encoders.items():
            if col in X_proc.columns:
                if isinstance(encoder, DateNormalizer) and encoder.min_date is None:
                    encoder.fit(X_proc[col])

                if isinstance(encoder, CostOfLivingEncoder):
                    feature_name = f"{col}_CostOfLiving"
                elif isinstance(encoder, MetroPopulationEncoder):
                    feature_name = f"{col}_MetroPopulation"
                elif isinstance(encoder, DateNormalizer):
                    feature_name = f"{col}_Normalized"
                else:
                    feature_name = f"{col}_Optional"

                X_proc[feature_name] = encoder.transform(X_proc[col])
                optional_feature_names.append(feature_name)

        missing_feats = [f for f in self.feature_names if f not in X_proc.columns]
        if missing_feats:
            for f in missing_feats:
                if f in X.columns:
                    X_proc[f] = X[f]

        all_feature_names = list(self.feature_names) + [
            f for f in optional_feature_names if f not in self.feature_names
        ]
        return X_proc[all_feature_names]

    def remove_outliers(
        self, df: pd.DataFrame, method: str = "iqr", threshold: float = 1.5
    ) -> Tuple[pd.DataFrame, int]:
        """Removes outliers from the dataframe based on target columns. Args: df (pd.DataFrame): Input data. method (str): Outlier method. threshold (float): IQR multiplier. Returns: Tuple[pd.DataFrame, int]: Tuple of filtered dataframe and number of rows removed."""
        if method != "iqr":
            raise NotImplementedError("Only IQR method is currently supported.")

        df_clean = df.copy()
        initial_len = len(df_clean)

        mask = pd.Series(True, index=df_clean.index)

        for target in self.targets:
            if target in df_clean.columns:
                q1 = df_clean[target].quantile(0.25)
                q3 = df_clean[target].quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr

                col_mask = (df_clean[target] >= lower_bound) & (df_clean[target] <= upper_bound)
                mask = mask & col_mask

        df_clean = df_clean[mask]
        removed_count = initial_len - len(df_clean)

        return df_clean, removed_count

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
        remove_outliers: bool,
        callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]],
    ) -> pd.DataFrame:
        """Prepare training data with optional outlier removal. Args: df (pd.DataFrame): Training data. remove_outliers (bool): Remove outliers. callback (Optional[Callable]): Progress callback. Returns: pd.DataFrame: Prepared DataFrame."""
        if remove_outliers:
            if callback:
                callback("Preprocessing: Removing outliers...", {"stage": "preprocess"})
            else:
                self.logger.info("Preprocessing: Removing outliers...")

            df, removed = self.remove_outliers(df)
            msg = f"Removed {removed} outlier rows."
            if callback:
                callback(msg, {"stage": "preprocess_result", "removed": removed})
            else:
                self.logger.info(msg)
        return df

    def _get_training_params(self, quantile: float, monotone_constraints: str) -> Dict[str, Any]:
        """Get training parameters for a specific quantile. Args: quantile (float): Quantile value. monotone_constraints (str): Monotonic constraints string. Returns: Dict[str, Any]: Training parameters."""
        hyperparams = self.model_config.get("hyperparameters", {})
        train_params_config = hyperparams.get(
            "training", {"objective": "reg:quantileerror", "tree_method": "hist", "verbosity": 0}
        )

        params = train_params_config.copy()
        params.update(
            {
                "quantile_alpha": quantile,
                "monotone_constraints": monotone_constraints,
            }
        )
        return params

    def _get_cv_params(self) -> Dict[str, Any]:
        """Get cross-validation parameters. Returns: Dict[str, Any]: CV parameters."""
        hyperparams = self.model_config.get("hyperparameters", {})
        return hyperparams.get(
            "cv",
            {
                "num_boost_round": 100,
                "nfold": 5,
                "early_stopping_rounds": 10,
                "verbose_eval": False,
            },
        )

    def _train_single_model(
        self,
        model_name: str,
        dtrain: xgb.DMatrix,
        params: Dict[str, Any],
        cv_params: Dict[str, Any],
        callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]],
    ) -> xgb.Booster:
        """Train a single quantile model. Args: model_name (str): Model name. dtrain (xgb.DMatrix): Training data. params (Dict[str, Any]): Training parameters. cv_params (Dict[str, Any]): CV parameters. callback (Optional[Callable]): Progress callback. Returns: xgb.Booster: Trained model."""
        if callback:
            callback(f"Training {model_name}...", {"stage": "start", "model_name": model_name})
        else:
            self.logger.info(f"Training {model_name}...")

        if callback:
            callback(f"Running Cross-Validation...", {"stage": "cv_start"})
        else:
            self.logger.debug(f"Running Cross-Validation for {model_name}...")

        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=cv_params.get("num_boost_round", 100),
            nfold=cv_params.get("nfold", 5),
            early_stopping_rounds=cv_params.get("early_stopping_rounds", 10),
            metrics={"quantile"},
            seed=42,
            verbose_eval=cv_params.get("verbose_eval", False),
        )

        metric_name = "test-quantile-mean"
        best_round, best_score = self._analyze_cv_results(cv_results, metric_name)

        if callback:
            data = {
                "stage": "cv_end",
                "model_name": model_name,
                "best_round": best_round,
                "best_score": best_score,
                "metric_name": metric_name,
            }
            callback(f"Best Round: {best_round}, Score: {best_score:.4f}", data)
        else:
            self.logger.info(
                f"  Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}"
            )

        model = xgb.train(params, dtrain, num_boost_round=best_round)
        return model

    def train(
        self,
        df: pd.DataFrame,
        callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
        remove_outliers: bool = False,
    ) -> None:
        """Trains the XGBoost models. Args: df (pd.DataFrame): Training data. callback (Optional[Callable]): Optional callback for status updates. remove_outliers (bool): If True, applies IQR outlier removal before training. Returns: None."""
        df = self._prepare_training_data(df, remove_outliers, callback)

        X = self._preprocess(df)
        weights = self.weighter.transform(df)

        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))

        cv_params = self._get_cv_params()

        for target in self.targets:
            y = df[target]
            dtrain = xgb.DMatrix(X, label=y, weight=weights)

            for q in self.quantiles:
                model_name = f"{target}_p{int(q*100)}"
                params = self._get_training_params(q, monotone_constraints)
                model = self._train_single_model(model_name, dtrain, params, cv_params, callback)
                self.models[model_name] = model

                dtrain = xgb.DMatrix(X, label=y, weight=weights)

                if callback:
                    callback(f"Running Cross-Validation...", {"stage": "cv_start"})
                else:
                    self.logger.debug(f"Running Cross-Validation for {model_name}...")

                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=cv_params_config.get("num_boost_round", 100),
                    nfold=cv_params_config.get("nfold", 5),
                    early_stopping_rounds=cv_params_config.get("early_stopping_rounds", 10),
                    metrics={"quantile"},
                    seed=42,
                    verbose_eval=cv_params_config.get("verbose_eval", False),
                )

                metric_name = "test-quantile-mean"
                best_round, best_score = self._analyze_cv_results(cv_results, metric_name)

                if callback:
                    data = {
                        "stage": "cv_end",
                        "model_name": model_name,
                        "best_round": best_round,
                        "best_score": best_score,
                        "metric_name": metric_name,
                    }
                    callback(f"Best Round: {best_round}, Score: {best_score:.4f}", data)
                else:
                    self.logger.info(
                        f"  Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}"
                    )

                model = xgb.train(params, dtrain, num_boost_round=best_round)
                self.models[model_name] = model

    def tune(
        self, df: pd.DataFrame, n_trials: int = 20, timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs Optuna optimization to find best hyperparameters and updates self.model_config. Args: df (pd.DataFrame): Training data. n_trials (int): Number of trials. timeout (Optional[int]): Timeout in seconds. Returns: Dict[str, Any]: Best parameters found."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X = self._preprocess(df)
        weights = self.weighter.transform(df)

        target = self.targets[0]
        y = df[target]
        q_tune = 0.5 if 0.5 in self.quantiles else self.quantiles[0]

        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))

        dtrain = xgb.DMatrix(X, label=y, weight=weights)

        def objective(trial: optuna.trial.Trial) -> float:
            params = {
                "objective": "reg:quantileerror",
                "tree_method": "hist",
                "verbosity": 0,
                "quantile_alpha": q_tune,
                "monotone_constraints": monotone_constraints,
                "eta": trial.suggest_float("eta", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=100,
                nfold=3,
                early_stopping_rounds=10,
                metrics={"quantile"},
                seed=42,
                verbose_eval=False,
            )

            score = cv_results["test-quantile-mean"].min()
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_params

        hyperparams = self.model_config.setdefault("hyperparameters", {})
        train_params = hyperparams.setdefault("training", {})
        train_params.update(best_params)

        return best_params

    def predict(self, X_input: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generates predictions for input data. Args: X_input (pd.DataFrame): Input data with features matching training columns. Returns: Dict[str, Dict[str, Any]]: A nested dictionary: {target: {quantile_key: predictions}}."""
        X_proc = self._preprocess(X_input)
        dtest = xgb.DMatrix(X_proc)

        results: Dict[str, Dict[str, Any]] = {}
        for target in self.targets:
            target_res = {}
            for q in self.quantiles:
                model_name = f"{target}_p{int(q*100)}"
                if model_name in self.models:
                    preds = self.models[model_name].predict(dtest)
                    target_res[f"p{int(q*100)}"] = preds
            results[target] = target_res

        return results

    @staticmethod
    def _analyze_cv_results(
        cv_results: pd.DataFrame, metric_name: str = "test-quantile-mean"
    ) -> Tuple[int, float]:
        """Analyzes cross-validation results to find optimal rounds and best score. Args: cv_results (pd.DataFrame): XGBoost CV results dataframe. metric_name (str): Name of the metric to analyze. Returns: Tuple[int, float]: Best round (1-based) and best score."""
        if metric_name not in cv_results.columns:
            raise ValueError(
                f"Metric {metric_name} not found in CV results columns: {cv_results.columns}"
            )

        best_round = cv_results[metric_name].argmin() + 1
        best_score = float(cv_results[metric_name].min())

        return best_round, best_score


# Backward Compatibility Aliases
SalaryForecaster = QuantileForecaster
