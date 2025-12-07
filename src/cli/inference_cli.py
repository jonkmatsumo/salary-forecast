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

logger = logging.getLogger(__name__)

def load_model(path: str) -> Any:
    if not os.path.exists(path):
        logger.error(f"Model file '{path}' not found.")
        sys.exit(1)
    
    with open(path, "rb") as f:
        return pickle.load(f)

def get_input(prompt: str, type_func: Callable[[str], Any] = str, valid_options: Optional[List[Any]] = None) -> Any:
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

def select_model(console: Console) -> str:
    models = glob.glob("*.pkl")
    if not models:
        console.print("[bold red]No model files (*.pkl) found in current directory.[/bold red]")
        console.print("Please run the training CLI first: python3 -m src.cli.train_cli")
        sys.exit(1)
        
    if len(models) == 1:
        console.print(f"[bold blue]Found one model: {models[0]}[/bold blue]")
        return models[0]
        
    console.print("\n[bold]Available Models:[/bold]")
    for i, m in enumerate(models):
        console.print(f"{i+1}. {m}")
        
    while True:
        try:
            choice = int(console.input("\nSelect a model (number): "))
            if 1 <= choice <= len(models):
                return models[choice-1]
            console.print("[red]Invalid selection.[/red]")
        except ValueError:
            console.print("[red]Please enter a number.[/red]")

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
    
    # Non-interactive arguments
    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--level", type=str, help="Candidate Level (e.g. E5)")
    parser.add_argument("--location", type=str, help="Candidate Location")
    parser.add_argument("--yoe", type=int, help="Years of Experience")
    parser.add_argument("--yac", type=int, help="Years at Company")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    if args.json:
        # If JSON output, suppress logs to stderr so stdout is clean JSON
        logging.basicConfig(level=log_level, stream=sys.stderr, format='%(message)s')
    else:
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    console = Console(stderr=args.json) # Print to stderr if json mode
    if not args.json:
        console.print("[bold green]Welcome to the Salary Forecasting CLI[/bold green]")
    
    # 1. Model Selection
    if args.model:
        model_path = args.model
    else:
        # Interactive selection
        model_path = select_model(console)

    if not args.json:
        logger.info(f"Loading model from: {model_path}")
    
    try:
        model = load_model(model_path)
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
