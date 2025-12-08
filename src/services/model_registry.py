import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Any
from src.model.model import SalaryForecaster
from mlflow.pyfunc import PythonModel
from src.utils.logger import get_logger

class SalaryForecasterWrapper(PythonModel):
    """Wrapper for MLflow persistence of SalaryForecaster."""
    def __init__(self, forecaster):
        self.forecaster = forecaster
    def predict(self, context, model_input: List[Any]):
        return self.forecaster.predict(model_input)
    def unwrap_python_model(self):
        return self.forecaster

class ModelRegistry:
    """Service for managing model persistence and retrieval via MLflow."""

    def __init__(self, experiment_name: str = "Salary_Forecast"):
        self.logger = get_logger(__name__)
        # Ensure experiment exists
        self.client = MlflowClient()
        self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        self.logger.debug(f"Initialized ModelRegistry with experiment: {experiment_name}")

    def list_models(self) -> List[Any]:
        """Lists successful runs that have a model artifact."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        # Return summary list/dataframe directly for now
        if len(runs) == 0:
            return []
        
        # Prepare a lightweight format for UI
        cols = ["run_id", "start_time"]
        if "metrics.cv_mean_score" in runs.columns:
            cols.append("metrics.cv_mean_score")
        elif "metrics.cv_score" in runs.columns:
             # Fallback if specific mean not logged? 
             pass
             
        # Extract available columns
        return runs[cols].to_dict('records')

    def load_model(self, run_id: str) -> SalaryForecaster:
        """Loads the 'model' artifact from the specified run."""
        model_uri = f"runs:/{run_id}/model"
        # Pyfunc load would require defining a PythonModel
        # Since we just pickled it as an sklearn flavor (or custom), 
        # we will use the native flavor if possible, or generic pickle.
        # For simplicity in this migration step, we assume we log it as sklearn/custom
        # NOT relying on pickle local load anymore.
        
        # NOTE: If we log with mlflow.sklearn.log_model(forecaster), 
        # it might fail because SalaryForecaster isn't an sklearn estimator.
        # But we can assume the trainer logs it correctly.
        
        self.logger.info(f"Loading model from run: {run_id}")
        return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().unwrap_python_model()

    def save_model(self, model: SalaryForecaster, run_name: str = None) -> None:
        """
        Deprecated for direct usage. 
        Models should be logged during training context.
        This generic save is kept for compatibility or manual saves.
        """
        if mlflow.active_run():
            mlflow.pyfunc.log_model(
                artifact_path="model", 
                python_model=model,
                pip_requirements=["xgboost", "pandas", "scikit-learn"]
            )
        else:
            # Start a dummy run if needed, or raise error
            pass
