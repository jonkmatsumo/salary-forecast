from typing import Any, List, Optional

import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient

from src.utils.env_loader import get_env_var
from src.utils.logger import get_logger
from src.xgboost.model import SalaryForecaster


def get_experiment_name() -> str:
    """Get MLflow experiment name from environment variable or default. Returns: str: Experiment name."""
    experiment_name = get_env_var("MLFLOW_EXPERIMENT_NAME", "AutoQuantile")
    return experiment_name


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

    def __init__(self, experiment_name: Optional[str] = None) -> None:
        self.logger = get_logger(__name__)

        if experiment_name is None:
            experiment_name = get_experiment_name()

        self.client = MlflowClient()
        self.experiment = mlflow.set_experiment(experiment_name)
        self.experiment_id = self.experiment.experiment_id if self.experiment else None
        self.logger.debug(f"Initialized ModelRegistry with experiment: {experiment_name}")

    def list_models(self) -> List[Any]:
        """List successful runs that have a model artifact from all experiments. Returns: List[Any]: List of run dictionaries."""
        try:
            try:
                all_experiments = self.client.search_experiments()
                experiment_ids = [
                    exp.experiment_id for exp in all_experiments if exp.lifecycle_stage != "deleted"
                ]
            except Exception as e:
                self.logger.warning(f"Could not get experiment list: {type(e).__name__}: {e}")
                experiment_ids = None

            runs = mlflow.search_runs(
                experiment_ids=experiment_ids,
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
            )
        except (AttributeError, ValueError) as e:
            if "'NoneType' object has no attribute 'copy'" in str(e):
                self.logger.warning(
                    "Encountered corrupted MLflow run data. "
                    "Some runs may have incomplete metadata. "
                    "Consider cleaning up the mlruns directory or removing corrupted run directories."
                )
                try:
                    runs = self._list_models_fallback()
                except Exception as fallback_error:
                    self.logger.error(
                        f"Fallback method also failed: {type(fallback_error).__name__}: {fallback_error}",
                        exc_info=True,
                    )
                    return []
            else:
                self.logger.error(f"Error listing models: {type(e).__name__}: {e}", exc_info=True)
                return []

        if len(runs) == 0:
            return []

        cols_to_keep = ["run_id", "start_time"]

        for c in runs.columns:
            if c.startswith("tags.") or c.startswith("metrics."):
                cols_to_keep.append(c)

        cols_to_keep = [c for c in cols_to_keep if c in runs.columns]

        return runs[cols_to_keep].to_dict("records")

    def _list_models_fallback(self) -> Any:
        """Fallback method to list models by manually iterating runs across all experiments. Returns: Any: DataFrame of runs."""
        from datetime import datetime

        import pandas as pd

        run_data = []
        try:
            try:
                all_experiments = self.client.search_experiments()
                experiment_ids = [
                    exp.experiment_id for exp in all_experiments if exp.lifecycle_stage != "deleted"
                ]
            except Exception as e:
                self.logger.warning(
                    f"Could not get experiment list in fallback: {type(e).__name__}: {e}"
                )
                experiment_ids = None

            try:
                runs = self.client.search_runs(
                    experiment_ids=experiment_ids, filter_string="status = 'FINISHED'"
                )
            except (AttributeError, ValueError) as search_error:
                if "'NoneType' object has no attribute 'copy'" in str(search_error):
                    self.logger.warning(
                        "Cannot list runs due to corrupted metadata. "
                        "Please manually remove corrupted run directories from mlruns/."
                    )
                    return pd.DataFrame()
                raise

            for run in runs:
                try:
                    run_id = (
                        run.info.run_id
                        if hasattr(run, "info") and hasattr(run.info, "run_id")
                        else "unknown"
                    )

                    run_dict = {
                        "run_id": run_id,
                        "start_time": (
                            datetime.fromtimestamp(run.info.start_time / 1000.0)
                            if hasattr(run, "info") and hasattr(run.info, "start_time")
                            else None
                        ),
                    }

                    if (
                        hasattr(run, "data")
                        and run.data
                        and hasattr(run.data, "tags")
                        and run.data.tags
                    ):
                        for key, value in run.data.tags.items():
                            run_dict[f"tags.{key}"] = value

                    if (
                        hasattr(run, "data")
                        and run.data
                        and hasattr(run.data, "metrics")
                        and run.data.metrics
                    ):
                        for key, value in run.data.metrics.items():
                            run_dict[f"metrics.{key}"] = value

                    run_data.append(run_dict)
                except Exception as run_error:
                    try:
                        run_id = (
                            run.info.run_id
                            if hasattr(run, "info") and hasattr(run.info, "run_id")
                            else "unknown"
                        )
                    except Exception:
                        run_id = "unknown"
                    self.logger.warning(
                        f"Skipping corrupted run {run_id}: {type(run_error).__name__}: {run_error}"
                    )
                    continue

            if not run_data:
                return pd.DataFrame()

            df = pd.DataFrame(run_data)
            if not df.empty:
                df = df.sort_values("start_time", ascending=False)

            return df
        except Exception as e:
            self.logger.error(f"Fallback method failed: {type(e).__name__}: {e}", exc_info=True)
            return pd.DataFrame()

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
                pip_requirements=["xgboost", "pandas", "scikit-learn"],
            )
        else:

            pass
