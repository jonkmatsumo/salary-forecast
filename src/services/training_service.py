import asyncio
import io
import threading
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.services.model_registry import SalaryForecasterWrapper, get_experiment_name
from src.utils.csv_validator import validate_csv
from src.utils.logger import get_logger
from src.utils.performance import (
    LLMCallTracker,
    PerformanceMetrics,
    get_llm_metrics_summary,
    get_metric_stats,
    set_global_llm_tracker,
    timing_decorator,
)
from src.xgboost.model import SalaryForecaster


class TrainingService:
    """Service for orchestrating model training and hyperparameter tuning."""

    def __init__(self):
        self.logger = get_logger(__name__)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._background_tasks: set = set()
        self.logger.debug("Initialized TrainingService")

    @timing_decorator(metric_name="training_total_time")
    def train_model(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        remove_outliers: bool = True,
        callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> SalaryForecaster:
        """Synchronous training (blocking). Args: data (pd.DataFrame): Training data. config (Dict[str, Any]): Required configuration dictionary. remove_outliers (bool): Remove outliers. callback (Optional[Callable]): Progress callback. Returns: SalaryForecaster: Trained model. Raises: ValueError: If config is missing or invalid."""
        if not config:
            raise ValueError(
                "Config is required. Generate config using WorkflowService first. "
                "See DESIGN_LLM_ONLY_CONFIG.md for migration guide."
            )
        forecaster = SalaryForecaster(config=config)
        if callback:
            callback("Starting training...", None)
        forecaster.train(data, callback=callback, remove_outliers=remove_outliers)
        return forecaster

    @timing_decorator(metric_name="tuning_total_time")
    def tune_model(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        n_trials: int = 20,
        callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Synchronous tuning (blocking). Args: data (pd.DataFrame): Training data. config (Dict[str, Any]): Required configuration dictionary. n_trials (int): Number of trials. callback (Optional[Callable]): Progress callback. Returns: Dict[str, Any]: Best hyperparameters. Raises: ValueError: If config is missing or invalid."""
        if not config:
            raise ValueError(
                "Config is required. Generate config using WorkflowService first. "
                "See DESIGN_LLM_ONLY_CONFIG.md for migration guide."
            )
        forecaster = SalaryForecaster(config=config)
        if callback:
            callback(f"Starting tuning with {n_trials} trials...", None)
        best_params = forecaster.tune(data, n_trials=n_trials)
        return best_params

    def start_training_async(
        self,
        data: pd.DataFrame,
        config: Dict[str, Any],
        remove_outliers: bool = True,
        do_tune: bool = False,
        n_trials: int = 20,
        additional_tag: Optional[str] = None,
        dataset_name: str = "Unknown",
    ) -> str:
        """Start training in a background asyncio task. Args: data (pd.DataFrame): Training data. config (Dict[str, Any]): Required configuration dictionary. remove_outliers (bool): Remove outliers. do_tune (bool): Run tuning. n_trials (int): Tuning trials. additional_tag (Optional[str]): Additional tag. dataset_name (str): Dataset name. Returns: str: Job ID. Raises: ValueError: If config is missing or invalid."""
        if not config:
            raise ValueError(
                "Config is required. Generate config using WorkflowService first. "
                "See DESIGN_LLM_ONLY_CONFIG.md for migration guide."
            )
        job_id = str(uuid.uuid4())

        with self._lock:
            self._jobs[job_id] = {
                "status": "QUEUED",
                "submitted_at": datetime.now(),
                "logs": [],
                "history": [],
                "scores": [],
                "result": None,
                "error": None,
            }

        # Run async task in background
        # Use asyncio.ensure_future to handle both existing and new event loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, schedule task
                task = asyncio.create_task(
                    self._run_async_job(
                        job_id,
                        data,
                        config,
                        remove_outliers,
                        do_tune,
                        n_trials,
                        additional_tag,
                        dataset_name,
                    )
                )
            else:
                # If loop exists but not running, create task
                task = loop.create_task(
                    self._run_async_job(
                        job_id,
                        data,
                        config,
                        remove_outliers,
                        do_tune,
                        n_trials,
                        additional_tag,
                        dataset_name,
                    )
                )
        except RuntimeError:
            # No event loop, create new one and run in thread
            import threading

            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                task = new_loop.create_task(
                    self._run_async_job(
                        job_id,
                        data,
                        config,
                        remove_outliers,
                        do_tune,
                        n_trials,
                        additional_tag,
                        dataset_name,
                    )
                )
                new_loop.run_until_complete(task)

            thread = threading.Thread(target=run_in_new_loop, daemon=True)
            thread.start()
            return job_id

        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

        return job_id

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status dictionary of a job. Args: job_id (str): Job identifier. Returns: Optional[Dict[str, Any]]: Job status dictionary or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def validate_csv_file(
        self, file_content: bytes, filename: str
    ) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
        """Validate and parse a CSV file. Args: file_content (bytes): CSV file content. filename (str): Original filename. Returns: Tuple[bool, Optional[str], Optional[pd.DataFrame]]: (is_valid, error_message, dataframe)."""
        try:
            file_buffer = io.BytesIO(file_content)
            is_valid, error_msg, df = validate_csv(file_buffer)

            if not is_valid:
                return False, error_msg, None

            return True, None, df
        except Exception as e:
            self.logger.error(f"CSV validation failed for {filename}: {e}", exc_info=True)
            return False, f"Failed to validate CSV file: {str(e)}", None

    def parse_csv_data(self, file_content: bytes) -> pd.DataFrame:
        """Parse CSV file content into a DataFrame. Args: file_content (bytes): CSV file content. Returns: pd.DataFrame: Parsed DataFrame. Raises: ValueError: If CSV cannot be parsed."""
        file_buffer = io.BytesIO(file_content)
        is_valid, error_msg, df = validate_csv(file_buffer)

        if not is_valid:
            raise ValueError(error_msg or "Invalid CSV file")

        if df is None:
            raise ValueError("Failed to parse CSV file")

        return df

    def get_training_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a training job suitable for API responses. Args: job_id (str): Job identifier. Returns: Optional[Dict[str, Any]]: Job summary or None if not found."""
        job_status = self.get_job_status(job_id)
        if job_status is None:
            return None

        summary = {
            "job_id": job_id,
            "status": job_status.get("status"),
            "submitted_at": job_status.get("submitted_at"),
            "completed_at": job_status.get("completed_at"),
            "run_id": job_status.get("run_id"),
        }

        if job_status.get("status") == "COMPLETED":
            summary["result"] = "Model trained successfully"
            if "run_id" in job_status:
                summary["run_id"] = job_status["run_id"]
        elif job_status.get("status") == "FAILED":
            summary["error"] = job_status.get("error")

        return summary

    async def _run_async_job(
        self,
        job_id: str,
        data: pd.DataFrame,
        config: Dict[str, Any],
        remove_outliers: bool,
        do_tune: bool,
        n_trials: int,
        additional_tag: Optional[str],
        dataset_name: str,
    ) -> None:
        """Internal async worker method. Args: job_id (str): Job identifier. data (pd.DataFrame): Training data. config (Dict[str, Any]): Required configuration dictionary. remove_outliers (bool): Remove outliers. do_tune (bool): Run tuning. n_trials (int): Tuning trials. additional_tag (Optional[str]): Additional tag. dataset_name (str): Dataset name. Returns: None."""
        loop = asyncio.get_event_loop()

        def _async_callback(msg: str, data: Optional[Dict[str, Any]] = None) -> None:
            """Thread-safe callback for training progress. Args: msg (str): Message. data (Optional[Dict[str, Any]]): Optional data. Returns: None."""
            with self._lock:
                if job_id in self._jobs:
                    self._jobs[job_id]["logs"].append(msg)
                    if data:
                        self._jobs[job_id]["history"].append(data)
                        if data.get("stage") == "cv_end":
                            score = data.get("best_score")
                            if score is not None:
                                self._jobs[job_id]["scores"].append(score)

                            try:
                                mlflow.log_metric("cv_score", score)
                            except Exception:
                                pass  # Ignore if no active run

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

            global_tracker = LLMCallTracker(
                model="aggregate", provider="mixed", global_tracking=False
            )
            set_global_llm_tracker(global_tracker)

            with mlflow.start_run(run_name=run_name) as run:

                mlflow.set_tags(
                    {
                        "model_type": "XGBoost",
                        "dataset_name": dataset_name,
                        "additional_tag": additional_tag if additional_tag else "N/A",
                    }
                )

                mlflow.log_params(
                    {
                        "remove_outliers": remove_outliers,
                        "do_tune": do_tune,
                        "n_trials": n_trials if do_tune else 0,
                        "data_rows": len(data),
                    }
                )

                forecaster = SalaryForecaster(config=config)

                if do_tune:
                    self.logger.info(f"Starting tuning for job {job_id}")
                    _async_callback(f"Starting tuning with {n_trials} trials...")
                    with PerformanceMetrics("tuning_total_time"):
                        # Run CPU-bound tuning in executor
                        best_params = await loop.run_in_executor(
                            None, lambda: forecaster.tune(data, n_trials=n_trials)
                        )
                    mlflow.log_params(best_params)
                    tuning_stats = get_metric_stats("tuning_total_time")
                    if tuning_stats:
                        mlflow.log_metric("tuning_total_time", tuning_stats["total"])
                        mlflow.log_metric("tuning_trials_count", n_trials)
                        if n_trials > 0:
                            mlflow.log_metric(
                                "tuning_avg_trial_time", tuning_stats["total"] / n_trials
                            )

                _async_callback("Starting training...")
                with PerformanceMetrics("training_total_time"):
                    # Run CPU-bound training in executor
                    await loop.run_in_executor(
                        None,
                        lambda: forecaster.train(
                            data, callback=_async_callback, remove_outliers=remove_outliers
                        ),
                    )
                training_stats = get_metric_stats("training_total_time")
                if training_stats:
                    mlflow.log_metric("training_total_time", training_stats["total"])

                preprocessing_stats = get_metric_stats("preprocessing_feature_encoding_time")
                if preprocessing_stats:
                    mlflow.log_metric("preprocessing_total_time", preprocessing_stats["total"])

                preprocessing_cleaning = get_metric_stats("preprocessing_data_cleaning_time")
                preprocessing_outlier = get_metric_stats("preprocessing_outlier_removal_time")
                if preprocessing_cleaning:
                    mlflow.log_metric(
                        "preprocessing_data_cleaning_time", preprocessing_cleaning["total"]
                    )
                if preprocessing_outlier:
                    mlflow.log_metric(
                        "preprocessing_outlier_removal_time", preprocessing_outlier["total"]
                    )

                llm_summary = get_llm_metrics_summary()
                if llm_summary:
                    mlflow.log_metric("llm_total_tokens", llm_summary["total_tokens"])
                    mlflow.log_metric("llm_total_cost", llm_summary["total_cost"])
                    mlflow.log_metric("llm_avg_latency", llm_summary["avg_latency"])
                    mlflow.log_metric("llm_call_count", llm_summary["call_count"])

                wrapper = SalaryForecasterWrapper(forecaster)
                mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model=wrapper,
                    pip_requirements=["xgboost", "pandas", "scikit-learn"],
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
