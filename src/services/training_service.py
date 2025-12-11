from typing import Dict, Any, Optional, Callable
import pandas as pd
from datetime import datetime
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import mlflow
from src.services.model_registry import SalaryForecasterWrapper, get_experiment_name
from src.xgboost.model import SalaryForecaster
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
                   callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
                   config: Optional[Dict[str, Any]] = None) -> SalaryForecaster:
        """Synchronous training (blocking). Args: data (pd.DataFrame): Training data. remove_outliers (bool): Remove outliers. callback (Optional[Callable]): Progress callback. config (Optional[Dict[str, Any]]): Optional config dict. Returns: SalaryForecaster: Trained model."""
        forecaster = SalaryForecaster(config=config)
        if callback:
            callback("Starting training...", None)
        forecaster.train(data, callback=callback, remove_outliers=remove_outliers)
        return forecaster

    def tune_model(self, 
                  data: pd.DataFrame, 
                  n_trials: int = 20, 
                  callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None) -> Dict[str, Any]:
        """Synchronous tuning (blocking). Args: data (pd.DataFrame): Training data. n_trials (int): Number of trials. callback (Optional[Callable]): Progress callback. Returns: Dict[str, Any]: Best hyperparameters."""
        forecaster = SalaryForecaster()
        if callback:
            callback(f"Starting tuning with {n_trials} trials...", None)
        best_params = forecaster.tune(data, n_trials=n_trials)
        return best_params



    def start_training_async(self, 
                           data: pd.DataFrame, 
                           remove_outliers: bool = True,
                           do_tune: bool = False,
                           n_trials: int = 20,
                           additional_tag: Optional[str] = None,
                           dataset_name: str = "Unknown") -> str:
        """Start training in a background thread. Args: data (pd.DataFrame): Training data. remove_outliers (bool): Remove outliers. do_tune (bool): Run tuning. n_trials (int): Tuning trials. additional_tag (Optional[str]): Additional tag. dataset_name (str): Dataset name. Returns: str: Job ID."""
        job_id = str(uuid.uuid4())
        
        with self._lock:
            self._jobs[job_id] = {
                "status": "QUEUED",
                "submitted_at": datetime.now(),
                "logs": [],
                "history": [], 
                "scores": [],
                "result": None,
                "error": None
            }
            
        self.executor.submit(
            self._run_async_job, 
            job_id, 
            data, 
            remove_outliers, 
            do_tune, 
            n_trials,
            additional_tag,
            dataset_name
        )
        
        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns the current status dictionary of a job."""
        with self._lock:
            return self._jobs.get(job_id)

    def _run_async_job(self, job_id: str, data: pd.DataFrame, remove_outliers: bool, do_tune: bool, n_trials: int, additional_tag: Optional[str], dataset_name: str) -> None:
        """Internal worker method. Args: job_id (str): Job identifier. data (pd.DataFrame): Training data. remove_outliers (bool): Remove outliers. do_tune (bool): Run tuning. n_trials (int): Tuning trials. additional_tag (Optional[str]): Additional tag. dataset_name (str): Dataset name. Returns: None."""
        

        def _async_callback(msg: str, data: Optional[Dict[str, Any]] = None) -> None:
            """Async callback for training progress. Args: msg (str): Message. data (Optional[Dict[str, Any]]): Optional data. Returns: None."""
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["logs"].append(msg)
                    if data:
                        self._jobs[job_id]["history"].append(data)
                        if data.get("stage") == "cv_end":
                            score = data.get("best_score")
                            self._jobs[job_id]["scores"].append(score)

                            try:
                                mlflow.log_metric("cv_score", score)
                            except:
                                pass # Ignore if no active run
                    
                    self._jobs[job_id]["last_update"] = datetime.now()

        try:
            with self._lock:
                self._jobs[job_id]["status"] = "RUNNING"
            
            self.logger.info(f"Starting async training job: {job_id}")

            experiment_name = get_experiment_name()
            mlflow.set_experiment(experiment_name)
            run_name = f"Training_{job_id}"
            if additional_tag:
                run_name = additional_tag
            
            with mlflow.start_run(run_name=run_name) as run:
                
                mlflow.set_tags({
                    "model_type": "XGBoost",
                    "dataset_name": dataset_name,
                    "additional_tag": additional_tag if additional_tag else "N/A"
                })

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
                

                wrapper = SalaryForecasterWrapper(forecaster)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    pip_requirements=["xgboost", "pandas", "scikit-learn"]
                )
                

                scores = self._jobs[job_id].get("scores", [])
                if scores:
                    mean_score = np.mean(scores)
                    mlflow.log_metric("cv_mean_score", mean_score)
                    self.logger.info(f"Job {job_id} finished. CV Mean Score: {mean_score:.4f}")
                

                
                with self._lock:
                    self._jobs[job_id]["status"] = "COMPLETED"
                    self._jobs[job_id]["result"] = forecaster
                    self._jobs[job_id]["run_id"] = run.info.run_id

                    self._jobs[job_id]["completed_at"] = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Training job {job_id} failed: {e}", exc_info=True)
            with self._lock:
                self._jobs[job_id]["status"] = "FAILED"
                self._jobs[job_id]["error"] = str(e)
                self._jobs[job_id]["completed_at"] = datetime.now()
