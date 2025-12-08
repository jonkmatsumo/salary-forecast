from typing import Dict, Any, Optional, Callable
import pandas as pd
from datetime import datetime
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
from src.model.model import SalaryForecaster
from src.utils.logger import get_logger

class TrainingService:
    """Service for orchestrating model training and hyperparameter tuning."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.logger.debug("Initialized TrainingService")

    def train_model(self, 
                   data: pd.DataFrame, 
                   remove_outliers: bool = True,
                   callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None) -> SalaryForecaster:
        """Synchronous training (blocking)."""
        forecaster = SalaryForecaster()
        if callback:
            callback("Starting training...", None)
        forecaster.train(data, callback=callback, remove_outliers=remove_outliers)
        return forecaster

    def tune_model(self, 
                  data: pd.DataFrame, 
                  n_trials: int = 20, 
                  callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None) -> Dict[str, Any]:
        """Synchronous tuning (blocking)."""
        forecaster = SalaryForecaster()
        if callback:
            callback(f"Starting tuning with {n_trials} trials...", None)
        best_params = forecaster.tune(data, n_trials=n_trials)
        return best_params

    # --- Async Methods ---

    def start_training_async(self, 
                           data: pd.DataFrame, 
                           remove_outliers: bool = True,
                           do_tune: bool = False,
                           n_trials: int = 20) -> str:
        """Starts training in a background thread and returns a Job ID."""
        job_id = str(uuid.uuid4())
        
        with self._lock:
            self._jobs[job_id] = {
                "status": "QUEUED",
                "submitted_at": datetime.now(),
                "logs": [],
                "logs": [],
                "history": [], 
                "scores": [], # Track CV scores for aggregation
                "result": None,
                "error": None
            }
            
        self.executor.submit(
            self._run_async_job, 
            job_id, 
            data, 
            remove_outliers, 
            do_tune, 
            n_trials
        )
        
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current status dictionary of a job."""
        with self._lock:
            return self._jobs.get(job_id)

    def _run_async_job(self, job_id: str, data: pd.DataFrame, remove_outliers: bool, do_tune: bool, n_trials: int):
        """Internal worker method."""
        import mlflow
        from src.services.model_registry import SalaryForecasterWrapper
        
        # Local callback to capture logs into the job state
        def _async_callback(msg: str, data: Optional[Dict[str, Any]] = None):
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["logs"].append(msg)
                    if data:
                        self._jobs[job_id]["history"].append(data)
                        # Log CV scores as metrics if available
                        if data.get("stage") == "cv_end":
                            score = data.get("best_score")
                            self._jobs[job_id]["scores"].append(score) # Track scores
                            try:
                                mlflow.log_metric("cv_score", score)
                            except:
                                pass # Ignore if no active run
                    
                    self._jobs[job_id]["last_update"] = datetime.now()

        try:
            with self._lock:
                self._jobs[job_id]["status"] = "RUNNING"
            
            self.logger.info(f"Starting async training job: {job_id}")

            # Start MLflow Run
            mlflow.set_experiment("Salary_Forecast")
            with mlflow.start_run(run_name=f"Training_{job_id}") as run:
                
                # Log Params
                mlflow.log_params({
                    "remove_outliers": remove_outliers,
                    "do_tune": do_tune,
                    "n_trials": n_trials if do_tune else 0,
                    "data_rows": len(data)
                })
                
                forecaster = SalaryForecaster()
                
                if do_tune:
                    self.logger.info(f"Starting tuning for job {job_id}")
                    _async_callback(f"Starting tuning with {n_trials} trials...")
                    best_params = forecaster.tune(data, n_trials=n_trials)
                    mlflow.log_params(best_params)
                    
                _async_callback("Starting training...")
                forecaster.train(data, callback=_async_callback, remove_outliers=remove_outliers)
                
                # Log Model
                wrapper = SalaryForecasterWrapper(forecaster)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    pip_requirements=["xgboost", "pandas", "scikit-learn"]
                )
                
                # Log final metrics if available
                scores = self._jobs[job_id].get("scores", [])
                if scores:
                    import numpy as np
                    mean_score = np.mean(scores)
                    mlflow.log_metric("cv_mean_score", mean_score)
                    self.logger.info(f"Job {job_id} finished. CV Mean Score: {mean_score:.4f}")
                
                # mlflow.log_metric("final_mae", ...) # Forecaster doesn't expose test metrics yet
                
                with self._lock:
                    self._jobs[job_id]["status"] = "COMPLETED"
                    self._jobs[job_id]["result"] = forecaster
                    self._jobs[job_id]["run_id"] = run.info.run_id # Store Run ID
                    self._jobs[job_id]["completed_at"] = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            with self._lock:
                self._jobs[job_id]["status"] = "FAILED"
                self._jobs[job_id]["error"] = str(e)
                self._jobs[job_id]["completed_at"] = datetime.now()
