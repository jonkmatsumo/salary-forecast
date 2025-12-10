import os
import argparse
import sys
import contextlib
import io
import traceback
from typing import Optional, Any

import pandas as pd
import mlflow
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.xgboost.model import SalaryForecaster
from src.utils.data_utils import load_data
from src.utils.config_loader import load_config
from src.services.model_registry import SalaryForecasterWrapper, get_experiment_name
from src.utils.logger import setup_logging

def train_workflow(csv_path: str, config_path: str, output_path: str, console: Any, do_tune: bool = False, num_trials: int = 20, remove_outliers: bool = False) -> None:
    """Execute the model training workflow. Args: csv_path (str): Training data CSV path. config_path (str): Config file path. output_path (str): Model output path. console (Any): Rich console. do_tune (bool): Run hyperparameter tuning. num_trials (int): Tuning trials. remove_outliers (bool): Remove outliers. Returns: None."""
    if not os.path.exists(csv_path):
        console.print(f"[bold red]Error: {csv_path} not found.[/bold red]")
        return

    if config_path and os.path.exists(config_path):
        load_config(config_path)

    status_text = Text("Status: Preparing...", style="bold blue")
    
    results_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Percentile", justify="right")
    results_table.add_column("Best Round", justify="right")
    results_table.add_column("Metric")
    results_table.add_column("Score", justify="right")

    output_group = Group(
        status_text,
        Text(""), 
        results_table
    )

    with Live(output_group, console=console, refresh_per_second=4, transient=False):
        status_text.plain = f"Status: Loading data from {csv_path}..."
        df = load_data(csv_path)
        
        status_text.plain = "Status: Starting training workflow..."
        status_text.plain = "Status: Initializing target cities..."
        
        with contextlib.redirect_stdout(io.StringIO()):
             forecaster = SalaryForecaster()
        
        if do_tune:
            status_text.plain = f"Status: Tuning hyperparameters (Trials={num_trials})..."
            best_params = forecaster.tune(df, n_trials=num_trials)
            console.print(f"[dim]Best Params: {best_params}[/dim]")
        
        status_text.plain = "Status: Starting training..."
        
        def console_callback(msg: str, data: Optional[dict] = None) -> None:
            if data and data.get("stage") == "start":
                model_name = data['model_name']
                status_text.plain = f"Status: Training {model_name}..."
            
            elif data and data.get("stage") == "cv_end":
                metric = data.get('metric_name', 'metric')
                best_round = str(data.get('best_round'))
                best_score = f"{data.get('best_score'):.4f}"
                model_name = data.get('model_name', 'Unknown')
                
                if '_p' in model_name:
                    parts = model_name.rsplit('_', 1)
                    component = parts[0]
                    percentile = parts[1]
                else:
                    component = model_name
                    percentile = "-"
                
                results_table.add_row(component, percentile, best_round, metric, best_score)
            
            elif data and data.get("stage") == "cv_start":
                pass
                
        forecaster.train(df, callback=console_callback, remove_outliers=remove_outliers)
        
        
        experiment_name = get_experiment_name()
        status_text.plain = f"Status: Logging model to MLflow (Experiment: {experiment_name})..."
        
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run() as run:
            # Log Params
            mlflow.log_params({
                "remove_outliers": remove_outliers,
                "do_tune": do_tune,
                "n_trials": num_trials if do_tune else 0,
                "data_rows": len(df)
            })
            

            
            # Log Model
            wrapper = SalaryForecasterWrapper(forecaster)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=wrapper,
                pip_requirements=["xgboost", "pandas", "scikit-learn"]
            )
            
            console.print(f"[dim]Run ID: {run.info.run_id}[/dim]")
            
        status_text.plain = "Status: Completed"
    
def main():
    setup_logging()
    console = Console()
    console.print("[bold green]Salary Forecasting Training CLI[/bold green]")
    
    parser = argparse.ArgumentParser(description="Train Salary Forecast Model")
    parser.add_argument("--csv", default="salaries-list.csv", help="Path to training CSV")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--output", default=None, help="Deprecated: Output path (now logs to MLflow)")
    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of tuning trials")
    parser.add_argument("--remove-outliers", action="store_true", help="Remove outliers using IQR before training")
    
    args = parser.parse_args()
    
    try:
        train_workflow(
            args.csv, 
            args.config, 
            args.output, 
            console, 
            do_tune=args.tune, 
            num_trials=args.num_trials,
            remove_outliers=args.remove_outliers
        )
        console.print(f"\n[bold green]Training workflow completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        traceback.print_exc()

if __name__ == "__main__":
    main()
