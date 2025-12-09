import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Any, Optional
from src.xgboost.model import SalaryForecaster
from mlflow.pyfunc import PythonModel
from src.utils.logger import get_logger

class SalaryForecasterWrapper(PythonModel):
    """Wrapper for MLflow persistence of SalaryForecaster."""
    def __init__(self, forecaster: Any) -> None:
        self.forecaster = forecaster
    def predict(self, context: Any, model_input: Any) -> Any:
        """Predict using wrapped forecaster. Args: context (Any): MLflow context. model_input (Any): Input data. Returns: Any: Predictions."""
        return self.forecaster.predict(model_input)
    def unwrap_python_model(self) -> Any:
        """Unwrap the Python model. Returns: Any: Unwrapped forecaster."""
        return self.forecaster

class ModelRegistry:
    """Service for managing model persistence and retrieval via MLflow."""

    def __init__(self, experiment_name: str = "Salary_Forecast") -> None:
        self.logger = get_logger(__name__)
        self.client = MlflowClient()
        self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id
        self.logger.debug(f"Initialized ModelRegistry with experiment: {experiment_name}")

    def list_models(self) -> List[Any]:
        """List successful runs that have a model artifact. Returns: List[Any]: List of run dictionaries."""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"]
        )
        if len(runs) == 0:
            return []
        
        cols_to_keep = ["run_id", "start_time"]
        
        for c in runs.columns:
            if c.startswith("tags.") or c.startswith("metrics."):
                cols_to_keep.append(c)
                
        # Filter to only existing columns
        cols_to_keep = [c for c in cols_to_keep if c in runs.columns]
        
        return runs[cols_to_keep].to_dict('records')

    def load_model(self, run_id: str) -> SalaryForecaster:
        """Load the 'model' artifact from the specified run. Args: run_id (str): MLflow run ID. Returns: SalaryForecaster: Loaded model."""
        model_uri = f"runs:/{run_id}/model"
        self.logger.info(f"Loading model from run: {run_id}")
        return mlflow.pyfunc.load_model(model_uri).unwrap_python_model().unwrap_python_model()

    def save_model(self, model: SalaryForecaster, run_name: Optional[str] = None) -> None:
        """Save model to MLflow. Args: model (SalaryForecaster): Model to save. run_name (Optional[str]): Run name. Returns: None."""
        if mlflow.active_run():
            mlflow.pyfunc.log_model(
                artifact_path="model", 
                python_model=model,
                pip_requirements=["xgboost", "pandas", "scikit-learn"]
            )
        else:

            pass
