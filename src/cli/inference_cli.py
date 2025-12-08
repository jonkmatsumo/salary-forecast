import pickle
import os
import pandas as pd
import sys
import glob
import plotext as plt
import logging
import argparse
from rich.console import Console
from rich.table import Table

from typing import Any, Callable, List, Optional, Union
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)

def load_model(path: str) -> Any:
    """Loads a pickled model object."""
    if not os.path.exists(path):
        logger.error(f"Model file '{path}' not found.")
        sys.exit(1)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def get_input(prompt: str, type_func: Callable[[str], Any] = str, valid_options: Optional[List[Any]] = None) -> Any:
    """Prompts user for input with type validation and optional allowed values."""
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input:
                continue
                
            val = type_func(user_input)
            
            if valid_options and val not in valid_options:
                print(f"Invalid option. Please choose from: {', '.join(map(str, valid_options))}")
                continue
                
            return val
        except ValueError:
            print(f"Invalid input. Please enter a valid {type_func.__name__}.")

def format_currency(val: float) -> str:
    return f"${val:,.0f}"

def collect_user_data() -> pd.DataFrame:
    """Interactive prompt for candidate details."""
    print("\n--- Enter Candidate Details ---")
    level = get_input("Level (e.g. E3, E4, E5, E6, E7): ", str, ["E3", "E4", "E5", "E6", "E7"])
    location = get_input("Location (e.g. New York, San Francisco): ", str)
    yoe = get_input("Years of Experience: ", int)
    yac = get_input("Years at Company: ", int)
    
    return pd.DataFrame([{
        "Level": level,
        "Location": location,
        "YearsOfExperience": yoe,
        "YearsAtCompany": yac
    }])

from src.services.model_registry import ModelRegistry

def select_model(console: Console, registry: ModelRegistry) -> str:
    """Interactive model selection from MLflow runs."""
    runs = registry.list_models()
    if not runs:
        console.print("[bold red]No trained models found in MLflow.[/bold red]")
        console.print("Please run the training CLI or App first.")
        sys.exit(1)
        
    # Simplify for CLI display
    console.print("\n[bold]Available Runs:[/bold]")
    run_map = []
    
    for i, r in enumerate(runs):
        run_id = r["run_id"]
        date_str = r['start_time'].strftime('%Y-%m-%d %H:%M')
        score = f"{r.get('metrics.cv_mean_score', 'N/A'):.4f}"
        console.print(f"{i+1}. {date_str} (Score: {score}) - ID: {run_id[:8]}")
        run_map.append(run_id)
        
    while True:
        try:
            choice = int(console.input("\nSelect a run (number): "))
            if 1 <= choice <= len(run_map):
                return run_map[choice-1]
            console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a number.[/red]")

def get_ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def get_ordinal_suffix(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

import json

def main():
    parser = argparse.ArgumentParser(description="Salary Forecasting Inference CLI")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    
    parser.add_argument("--run-id", type=str, help="MLflow Run ID")
    parser.add_argument("--level", type=str, help="Candidate Level (e.g. E5)")
    parser.add_argument("--location", type=str, help="Candidate Location")
    parser.add_argument("--yoe", type=int, help="Years of Experience")
    parser.add_argument("--yac", type=int, help="Years at Company")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARNING
    if args.json:
        # For JSON output, we write logs to stderr to keep stdout clean for JSON
        logging.basicConfig(level=log_level, stream=sys.stderr, format='%(message)s', force=True)
    else:
        setup_logging(level=log_level)

    console = Console(stderr=args.json)
    if not args.json:
        console.print("[bold green]Welcome to the Salary Forecasting CLI[/bold green]")
    
    registry = ModelRegistry()
    
    # 1. Model Selection
    if args.run_id:
        run_id = args.run_id
    else:
        # Interactive selection
        run_id = select_model(console, registry)

    if not args.json:
        logger.info(f"Loading model run: {run_id}")
    
    try:
        model = registry.load_model(run_id)
    except Exception as e:
        if args.json:
             print(json.dumps({"error": f"Failed to load model: {e}"}))
             sys.exit(1)
        else:
             logger.error(f"Failed to load model: {e}")
             sys.exit(1)
    
    # 2. Input Collection
    # Check if we have enough args for non-interactive mode
    non_interactive = all([args.level, args.location, args.yoe is not None, args.yac is not None])
    
    if non_interactive:
        input_df = pd.DataFrame([{
            "Level": args.level,
            "Location": args.location,
            "YearsOfExperience": args.yoe,
            "YearsAtCompany": args.yac
        }])
        run_once = True
    else:
        # Partially supplied args? Warn or Error?
        # If some are supplied but not all, we could ask for the rest, but simpler to just enforce all-or-nothing for automation.
        if any([args.level, args.location, args.yoe, args.yac]):
             console.print("[bold red]Error: For non-interactive mode, you must supply --level, --location, --yoe, and --yac.[/bold red]")
             sys.exit(1)
             
        run_once = False
        
    # Main Loop (runs once if non-interactive)
    while True:
        try:
            if not non_interactive:
                input_df = collect_user_data()
            
            if not args.json:
                logger.info("Calculating prediction...")
            
            results = model.predict(input_df)
            
            # Format Results
            if args.json:
                # Convert results to clean dict
                json_out = {}
                for target, preds in results.items():
                    # preds is a dict of arrays, need to convert arrays to lists/floats
                    target_out = {}
                    for q_key, val in preds.items():
                         target_out[q_key] = float(val[0])
                    json_out[target] = target_out
                print(json.dumps(json_out, indent=2))
            else:
                # Viz and Table
                # Dynamically add columns based on model quantiles
                quantiles = sorted(model.quantiles)
                quantile_labels = [get_ordinal_suffix(int(q * 100)) for q in quantiles]
    
                # Visualization
                logger.info("Visualizing Forecast...")
                plt.clear_figure()
                plt.title("Salary Forecast by Quantile")
                plt.xlabel("Quantile")
                plt.ylabel("Amount ($)")
                plt.theme("pro")
                
                for target, preds in results.items():
                    y_values = []
                    for q in quantiles:
                        key = f"p{int(q*100)}"
                        y_values.append(preds.get(key, [0])[0])
                    
                    plt.plot(range(len(quantiles)), y_values, label=target)
                
                plt.xticks(range(len(quantiles)), quantile_labels)
                
                plt.show()
    
                table = Table(title="Prediction Results")
                table.add_column("Component", style="cyan", no_wrap=True)
                
                for q_label in quantile_labels:
                    table.add_column(f"{q_label} Percentile", style="magenta")
                
                for target, preds in results.items():
                    row = [target]
                    for q in quantiles:
                        key = f"p{int(q*100)}"
                        val = preds.get(key, [0])[0]
                        row.append(format_currency(val))
                    table.add_row(*row)
                    
                console.print(table)
            
            if run_once:
                break
                
            cont = input("\nForecast another? (y/n): ").strip().lower()
            if cont != 'y':
                console.print("[bold]Goodbye![/bold]")
                break
                
        except KeyboardInterrupt:
            if not args.json:
                console.print("\n[bold]Goodbye![/bold]")
            break
        except Exception as e:
            if args.json:
                print(json.dumps({"error": str(e)}))
                sys.exit(1)
            else:
                logger.error(f"An error occurred: {e}")
                # Don't break loop in interactive mode unless fatal?
                if run_once:
                    sys.exit(1)

if __name__ == "__main__":
    main()
