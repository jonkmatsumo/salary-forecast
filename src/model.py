import xgboost as xgb
import pandas as pd
import numpy as np
from .preprocessing import LevelEncoder, LocationEncoder, SampleWeighter

class SalaryForecaster:
    def __init__(self):
        self.models = {}
        self.targets = ["BaseSalary", "Stock", "Bonus", "TotalComp"]
        self.quantiles = [0.25, 0.50, 0.75]
        
        self.level_encoder = LevelEncoder()
        self.loc_encoder = LocationEncoder()
        self.weighter = SampleWeighter(k=1.0)
        
    def _preprocess(self, X):
        X_proc = X.copy()
        X_proc["Level_Enc"] = self.level_encoder.transform(X["Level"])
        X_proc["Location_Enc"] = self.loc_encoder.transform(X["Location"])
        
        # Select features for model
        features = ["Level_Enc", "Location_Enc", "YearsOfExperience", "YearsAtCompany"]
        return X_proc[features]

    def train(self, df):
        X = self._preprocess(df)
        weights = self.weighter.transform(df["Date"])
        
        # Monotonic constraints
        # 1: increasing, -1: decreasing, 0: no constraint
        # Features: [Level_Enc, Location_Enc, YearsOfExperience, YearsAtCompany]
        # XGBoost expects a tuple
        monotone_constraints = "(1, 0, 1, 0)"
        
        for target in self.targets:
            y = df[target]
            
            for q in self.quantiles:
                model_name = f"{target}_p{int(q*100)}"
                print(f"Training {model_name}...")
                
                # XGBoost Quantile Regression parameters
                params = {
                    "objective": "reg:quantileerror",
                    "quantile_alpha": q,
                    "monotone_constraints": monotone_constraints,
                    "verbosity": 0,
                    "tree_method": "hist" # Often faster
                }
                
                dtrain = xgb.DMatrix(X, label=y, weight=weights)
                model = xgb.train(params, dtrain, num_boost_round=100)
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
