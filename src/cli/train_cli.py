import os
import pickle
import pandas as pd
from rich.console import Console
from src.model.model import SalaryForecaster
from src.utils.data_utils import load_data
from src.utils.config_loader import load_config

def get_input(console, prompt, default=None):
    prompt_str = f"{prompt} [default: {default}]: " if default else f"{prompt}: "
    user_input = console.input(prompt_str).strip()
    return user_input if user_input else default

def train_workflow(csv_path, config_path, output_path, console):
    if not os.path.exists(csv_path):
        console.print(f"[bold red]Error: {csv_path} not found.[/bold red]")
        return

    # Load config if provided
    if config_path and os.path.exists(config_path):
        load_config(config_path)

    # Create layout elements
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text
    from rich.console import Group
    from rich import box
    import contextlib
    import io

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
        
        status_text.plain = "Status: Starting training..."
        
        # Callback to handle rich output
        def console_callback(msg, data=None):
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
                # Maybe too fast to show "Running CV" for each?
                # We can update status if we want
                pass
                
        forecaster.train(df, callback=console_callback)
        
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
    
    csv_path = get_input(console, "Input CSV path", "salaries-list.csv")
    config_path = get_input(console, "Config JSON path", "config.json")
    output_path = get_input(console, "Output model path", "salary_model.pkl")
    
    try:
        train_workflow(csv_path, config_path, output_path, console)
        console.print(f"\n[bold green]Training workflow completed![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
