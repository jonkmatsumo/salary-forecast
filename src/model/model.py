import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
from src.model.preprocessing import LevelEncoder, LocationEncoder, SampleWeighter
from src.utils.config_loader import get_config


class SalaryForecaster:
    def __init__(self, config=None):
        self.models = {}
        # Use provided config or load from disk
        if config is None:
            config = get_config()
        
        model_config = config["model"]
        self.model_config = model_config
        
        self.targets = model_config["targets"]
        self.quantiles = model_config["quantiles"]
        
        self.level_encoder = LevelEncoder()
        self.loc_encoder = LocationEncoder()
        
        # Use k from config or default to 1.0 if not present
        k = model_config.get("sample_weight_k", 1.0)
        self.weighter = SampleWeighter(k=k)
        
        self.features_config = model_config["features"]
        self.feature_names = [f["name"] for f in self.features_config]
        
    def _preprocess(self, X):
        X_proc = X.copy()
        X_proc["Level_Enc"] = self.level_encoder.transform(X["Level"])
        X_proc["Location_Enc"] = self.loc_encoder.transform(X["Location"])
        
        # Select features for model based on config
        # Note: Some features might be raw columns (YearsOfExperience) and some might be engineered (Level_Enc)
        # The config lists the final feature names.
        # We need to ensure X_proc has all of them.
        # Assuming input X has "YearsOfExperience" and "YearsAtCompany" already.
        
        return X_proc[self.feature_names]

    def train(self, df, callback=None):
        """
        Trains the XGBoost models.
        
        Args:
            df (pd.DataFrame): Training data.
            callback (callable, optional): function(status_msg, result_data=None).
                                          status_msg is a string or formatted object.
                                          result_data is a dict with extra info (like cv scores).
        """
        X = self._preprocess(df)
        weights = self.weighter.transform(df["Date"])
        
        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))
        return constraints, monotone_constraints

    def remove_outliers(self, df, method="iqr", threshold=1.5):
        """
        Removes outliers from the dataframe based on target columns.
        
        Args:
            df (pd.DataFrame): Input data.
            method (str): "iqr" or "zscore" (only iqr implemented for now).
            threshold (float): Multiplier for IQR (typically 1.5).
            
        Returns:
            pd.DataFrame: Filtered dataframe.
            int: Number of rows removed.
        """
        if method != "iqr":
            raise NotImplementedError("Only IQR method is currently supported.")
            
        df_clean = df.copy()
        initial_len = len(df_clean)
        
        # Calculate mask for all targets
        mask = pd.Series(True, index=df_clean.index)
        
        for target in self.targets:
            if target in df_clean.columns:
                q1 = df_clean[target].quantile(0.25)
                q3 = df_clean[target].quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                
                # Update mask: Keep row if it's within bounds for THIS target
                # We want to remove row if ANY target is outlier? Or all?
                # Usually if ANY target is bad, the row is suspicious.
                col_mask = (df_clean[target] >= lower_bound) & (df_clean[target] <= upper_bound)
                mask = mask & col_mask
                
        df_clean = df_clean[mask]
        removed_count = initial_len - len(df_clean)
        
        return df_clean, removed_count

    def train(self, df, callback=None, remove_outliers=False):
        """
        Trains the XGBoost models.
        
        Args:
            df (pd.DataFrame): Training data.
            callback (callable, optional): function(status_msg, result_data=None).
            remove_outliers (bool): If True, applies IQR outlier removal before training.
        """
        if remove_outliers:
            if callback:
                callback("Preprocessing: Removing outliers...", {"stage": "preprocess"})
            else:
                print("Preprocessing: Removing outliers...")
                
            df, removed = self.remove_outliers(df)
            msg = f"Removed {removed} outlier rows."
            if callback:
                callback(msg, {"stage": "preprocess_result", "removed": removed})
            else:
                print(msg)
                
        X = self._preprocess(df)
        weights = self.weighter.transform(df["Date"])
        
        # Monotonic constraints
        # Construct tuple string like "(1, 0, 1, 0)" based on config order
        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))
        
        # Get defaults or user config for hyperparams
        hyperparams = self.model_config.get("hyperparameters", {})
        train_params_config = hyperparams.get("training", {
            "objective": "reg:quantileerror",
            "tree_method": "hist",
            "verbosity": 0
        })
        cv_params_config = hyperparams.get("cv", {
            "num_boost_round": 100,
            "nfold": 5,
            "early_stopping_rounds": 10,
            "verbose_eval": False
        })

        for target in self.targets:
            y = df[target]
            
            for q in self.quantiles:
                model_name = f"{target}_p{int(q*100)}"
                
                if callback:
                    callback(f"Training {model_name}...", {"stage": "start", "model_name": model_name})
                else:
                    print(f"Training {model_name}...")
                
                # XGBoost Quantile Regression parameters
                # Merge dynamic params with config params
                params = train_params_config.copy()
                params.update({
                    "quantile_alpha": q,
                    "monotone_constraints": monotone_constraints,
                })
                
                dtrain = xgb.DMatrix(X, label=y, weight=weights)
                
                # Cross-validation
                if callback:
                    callback(f"Running Cross-Validation...", {"stage": "cv_start"})
                
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=cv_params_config.get("num_boost_round", 100),
                    nfold=cv_params_config.get("nfold", 5),
                    early_stopping_rounds=cv_params_config.get("early_stopping_rounds", 10),
                    metrics={'quantile'}, # Use quantile error metric
                    seed=42,
                    verbose_eval=cv_params_config.get("verbose_eval", False)
                )
                
                # Analyze results
                metric_name = 'test-quantile-mean'
                best_round, best_score = self._analyze_cv_results(cv_results, metric_name)
                
                if callback:
                    data = {
                        "stage": "cv_end",
                        "model_name": model_name,
                        "best_round": best_round,
                        "best_score": best_score,
                        "metric_name": metric_name
                    }
                    callback(f"Best Round: {best_round}, Score: {best_score:.4f}", data)
                else:
                    print(f"  Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}")

                model = xgb.train(params, dtrain, num_boost_round=best_round)
                self.models[model_name] = model

    def tune(self, df, n_trials=20, timeout=None):
        """
        Runs Optuna optimization to find best hyperparameters.
        Updates self.model_config with the best parameters found.
        
        Args:
            df (pd.DataFrame): Training data.
            n_trials (int): Number of trials.
            timeout (int): Timeout in seconds.
            
        Returns:
            dict: Best parameters found.
        """
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        X = self._preprocess(df)
        weights = self.weighter.transform(df["Date"])
        
        # Tune on the first target and median (or first) quantile
        target = self.targets[0]
        y = df[target]
        q_tune = 0.5 if 0.5 in self.quantiles else self.quantiles[0]
        
        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))
        
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        
        def objective(trial):
            params = {
                "objective": "reg:quantileerror",
                "tree_method": "hist",
                "verbosity": 0,
                "quantile_alpha": q_tune,
                "monotone_constraints": monotone_constraints,
                
                # Search Space
                "eta": trial.suggest_float("eta", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "alpha": trial.suggest_float("alpha", 0.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0)
            }
            
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=100,
                nfold=3,
                early_stopping_rounds=10,
                metrics={'quantile'},
                seed=42,
                verbose_eval=False
            )
            
            score = cv_results['test-quantile-mean'].min()
            return score
            
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        best_params = study.best_params
        
        # Update config
        hyperparams = self.model_config.setdefault("hyperparameters", {})
        train_params = hyperparams.setdefault("training", {})
        train_params.update(best_params)
        
        return best_params
                
    def predict(self, X_input):
        """
        Returns a dictionary of DataFrames or a structured result.
        """
        X_proc = self._preprocess(X_input)
        dtest = xgb.DMatrix(X_proc)
        
        results = {}
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
    def _analyze_cv_results(cv_results: pd.DataFrame, metric_name: str = 'test-quantile-mean'):
        """
        Analyzes cross-validation results to find the optimal number of rounds and the best score.
        """
        if metric_name not in cv_results.columns:
            raise ValueError(f"Metric {metric_name} not found in CV results columns: {cv_results.columns}")
            
        best_round = cv_results[metric_name].argmin() + 1
        best_score = cv_results[metric_name].min()
        
        return best_round, best_score
