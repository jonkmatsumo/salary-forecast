import os
import pickle
import pandas as pd
import numpy as np
import contextlib
import io
import traceback
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from src.model.model import SalaryForecaster
from src.utils.data_utils import load_data
from src.utils.config_loader import load_config


import argparse
import sys
from typing import Optional, Any

def train_workflow(csv_path: str, config_path: str, output_path: str, console: Any, do_tune: bool = False, num_trials: int = 20, remove_outliers: bool = False) -> None:
    if not os.path.exists(csv_path):
        console.print(f"[bold red]Error: {csv_path} not found.[/bold red]")
        return

    # Load config if provided
    if config_path and os.path.exists(config_path):
        load_config(config_path)

    # Create layout elements
    status_text = Text("Status: Preparing...", style="bold blue")
    
    results_table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    results_table.add_column("Component", style="cyan")
    results_table.add_column("Percentile", justify="right")
    results_table.add_column("Best Round", justify="right")
    results_table.add_column("Metric")
    results_table.add_column("Score", justify="right")

    # Group them
    output_group = Group(
        status_text,
        Text(""), # spacer
        results_table
    )

    with Live(output_group, console=console, refresh_per_second=4, transient=False):
        status_text.plain = f"Status: Loading data from {csv_path}..."
        df = load_data(csv_path)
        
        status_text.plain = "Status: Starting training workflow..."
        
        status_text.plain = "Status: Initializing target cities..."
        # Suppress the internal print from geo_utils
        with contextlib.redirect_stdout(io.StringIO()):
             forecaster = SalaryForecaster()
        
        # Tuning Phase
        if do_tune:
            status_text.plain = f"Status: Tuning hyperparameters (Trials={num_trials})..."
            best_params = forecaster.tune(df, n_trials=num_trials)
            console.print(f"[dim]Best Params: {best_params}[/dim]")
        
        status_text.plain = "Status: Starting training..."
        
        # Callback to handle rich output
        def console_callback(msg: str, data: Optional[dict] = None) -> None:
            if data and data.get("stage") == "start":
                model_name = data['model_name']
                status_text.plain = f"Status: Training {model_name}..."
            
            elif data and data.get("stage") == "cv_end":
                metric = data.get('metric_name', 'metric')
                best_round = str(data.get('best_round'))
                best_score = f"{data.get('best_score'):.4f}"
                model_name = data.get('model_name', 'Unknown')
                
                # Parse component/percentile
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
        
        status_text.plain = f"Status: Saving model to {output_path}..."
        with open(output_path, "wb") as f:
            pickle.dump(forecaster, f)
            
        status_text.plain = "Status: Completed"
    
    # Simple inference check
    console.print("\n[bold]Running sample inference...[/bold]")
    sample_input = pd.DataFrame([{
        "Level": "E4",
        "Location": "New York",
        "YearsOfExperience": 3,
        "YearsAtCompany": 0
    }])
    
    prediction = forecaster.predict(sample_input)
    console.print("Prediction for E4 New Hire in NY (3 YOE):")
    for target, preds in prediction.items():
        res_str = f"  {target}: "
        parts = []
        for q in sorted(forecaster.quantiles):
            key = f"p{int(q*100)}"
            val = preds[key][0]
            # Simple text output for inference check is fine
            parts.append(f"P{int(q*100)}={val:,.0f}")
        console.print(res_str + ", ".join(parts))

def main():
    console = Console()
    console.print("[bold green]Salary Forecasting Training CLI[/bold green]")
    
    parser = argparse.ArgumentParser(description="Train Salary Forecast Model")
    parser.add_argument("--csv", default="salaries-list.csv", help="Path to training CSV")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--output", default="salary_model.pkl", help="Path to save model")
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
