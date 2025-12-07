import xgboost as xgb
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

    def train(self, df, console=None):
        X = self._preprocess(df)
        weights = self.weighter.transform(df["Date"])
        
        # Monotonic constraints
        # Construct tuple string like "(1, 0, 1, 0)" based on config order
        constraints = [f["monotone_constraint"] for f in self.features_config]
        monotone_constraints = str(tuple(constraints))
        
        for target in self.targets:
            y = df[target]
            
            for q in self.quantiles:
                model_name = f"{target}_p{int(q*100)}"
                if console:
                    console.print(f"Training [bold]{model_name}[/bold]...")
                else:
                    print(f"Training {model_name}...")
                
                # XGBoost Quantile Regression parameters
                params = {
                    "objective": "reg:quantileerror",
                    "quantile_alpha": q,
                    "monotone_constraints": monotone_constraints,
                    "verbosity": 0,
                    "tree_method": "hist"
                }
                
                dtrain = xgb.DMatrix(X, label=y, weight=weights)
                
                # Cross-validation
                if console:
                    console.print(f"  Running Cross-Validation for {model_name}...")
                
                cv_results = xgb.cv(
                    params,
                    dtrain,
                    num_boost_round=100,
                    nfold=5,
                    early_stopping_rounds=10,
                    metrics={'quantile'}, # Use quantile error metric
                    seed=42,
                    verbose_eval=False
                )
                
                # Analyze results
                # Metric name in cv_results will be test-quantile-mean
                metric_name = 'test-quantile-mean'
                best_round = cv_results[metric_name].argmin() + 1
                best_score = cv_results[metric_name].min()
                
                if console:
                    console.print(f"  [cyan]Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}[/cyan]")
                    console.print(f"  [dim]Training final model with {best_round} rounds...[/dim]")
                else:
                    print(f"  Optimal rounds: {best_round}, Best {metric_name}: {best_score:.4f}")

                model = xgb.train(params, dtrain, num_boost_round=best_round)
                self.models[model_name] = model
                
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
