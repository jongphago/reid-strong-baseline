# encoding: utf-8
"""
W&B Sweep launcher for Re-Identification

This script creates a Weights & Biases sweep for hyperparameter tuning.
For each sweep trial, it launches the regular train.py script with parameter overrides.

Usage:
  python tools/sweep.py \
      --config_file configs/baseline.yml \
      --sweep-config sweeps/sweep_test.yaml

Optional flags:
  --count           Number of sweep runs to execute
  --method          Sweep method: grid|random|bayes
  --sweep_id        Existing W&B sweep ID to resume
  --sweep-config    Path to YAML file containing sweep settings
"""

import argparse
import os
import shlex
import subprocess
import sys

import wandb
import yaml


def load_sweep_config(sweep_config_path):
    """Load sweep configuration from YAML file"""
    if not sweep_config_path:
        return {}
    
    try:
        with open(sweep_config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"WARNING: Failed to load sweep config '{sweep_config_path}': {e}")
        return {}


def load_wandb_project_entity(config_path):
    """Extract W&B project and entity from config file"""
    project = os.getenv("WANDB_PROJECT", "reid-strong-baseline")
    entity = os.getenv("WANDB_ENTITY", "")
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg, dict) and "WANDB" in cfg:
            wandb_cfg = cfg["WANDB"]
            if isinstance(wandb_cfg, dict):
                project = wandb_cfg.get("PROJECT", project)
                entity = wandb_cfg.get("ENTITY", entity)
    except Exception:
        pass
    
    return project, entity if entity else None


def as_list(value):
    """Convert value to list"""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [s.strip() for s in value.split(",") if s.strip()]
    return [value]


def main():
    parser = argparse.ArgumentParser(
        "reid_sweep",
        description="W&B Sweep for Re-Identification hyperparameter tuning"
    )
    
    # Required arguments
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to base config file (e.g., configs/baseline.yml)"
    )
    
    # Sweep configuration
    parser.add_argument(
        "--sweep-config",
        type=str,
        default=None,
        help="Path to sweep configuration YAML file"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of sweep runs to execute"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["grid", "random", "bayes"],
        default=None,
        help="Sweep method (default: from config or 'random')"
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="Existing W&B sweep ID to resume"
    )
    
    # Hyperparameter values (CLI overrides)
    parser.add_argument(
        "--lr_values",
        type=str,
        default=None,
        help="Comma-separated list of learning rates"
    )
    parser.add_argument(
        "--wd_values",
        type=str,
        default=None,
        help="Comma-separated list of weight decay values"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config_file):
        print(f"ERROR: Config file not found: {args.config_file}")
        sys.exit(1)
    
    # Load sweep configuration from YAML
    sweep_cfg = load_sweep_config(args.sweep_config)
    
    # Extract W&B project and entity
    project, entity = load_wandb_project_entity(args.config_file)
    
    # Prepare sweep configuration
    method = args.method or sweep_cfg.get("method", "random")
    name = sweep_cfg.get("name", "reid_sweep")
    metric = sweep_cfg.get("metric", {"name": "val/mAP", "goal": "maximize"})
    
    # Get parameter values (CLI overrides YAML)
    cfg_params = sweep_cfg.get("parameters", {})
    
    def process_param(param_value, default_value=None, param_type=float):
        """Process parameter value - supports both list and distribution formats"""
        if param_value is None:
            return default_value
        
        # If it's a dictionary with min/max/distribution (continuous distribution)
        if isinstance(param_value, dict):
            # W&B format for continuous distributions
            result = {}
            if "min" in param_value and "max" in param_value:
                result["min"] = param_type(param_value["min"])
                result["max"] = param_type(param_value["max"])
                if "distribution" in param_value:
                    result["distribution"] = param_value["distribution"]
                return result
            # If it has "values" key (discrete values in dict format)
            elif "values" in param_value:
                return {"values": [param_type(v) for v in param_value["values"]]}
        
        # If it's a list (discrete values)
        if isinstance(param_value, list):
            return {"values": [param_type(v) for v in param_value]}
        
        # Single value
        return {"values": [param_type(param_value)]}
    
    lr_values = process_param(
        as_list(args.lr_values) if args.lr_values else cfg_params.get("base_lr"),
        {"values": [0.00035]},
        float
    )
    wd_values = process_param(
        as_list(args.wd_values) if args.wd_values else cfg_params.get("weight_decay"),
        {"values": [0.0005]},
        float
    )
    
    # Build sweep configuration
    sweep_config = {
        "name": name,
        "method": method,
        "metric": metric,
        "parameters": {
            "base_lr": lr_values,
            "weight_decay": wd_values,
        }
    }
    
    # Add optional parameters if specified
    warmup_factor_param = process_param(cfg_params.get("warmup_factor"), None, float)
    if warmup_factor_param:
        sweep_config["parameters"]["warmup_factor"] = warmup_factor_param
    
    warmup_iters_param = process_param(cfg_params.get("warmup_iters"), None, int)
    if warmup_iters_param:
        sweep_config["parameters"]["warmup_iters"] = warmup_iters_param
    
    margin_param = process_param(cfg_params.get("margin"), None, float)
    if margin_param:
        sweep_config["parameters"]["margin"] = margin_param
    
    batch_size_param = process_param(cfg_params.get("ims_per_batch"), None, int)
    if batch_size_param:
        sweep_config["parameters"]["ims_per_batch"] = batch_size_param
    
    max_epochs_param = process_param(cfg_params.get("max_epochs"), None, int)
    if max_epochs_param:
        sweep_config["parameters"]["max_epochs"] = max_epochs_param
    
    # Add early termination if specified
    if sweep_cfg.get("early_terminate"):
        sweep_config["early_terminate"] = sweep_cfg["early_terminate"]
    
    # Create or resume sweep
    if args.sweep_id:
        sweep_id = args.sweep_id
        print(f"Resuming existing sweep: {sweep_id}")
    else:
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        print(f"Created new sweep: {sweep_id}")
    
    # Determine base results directory
    results_dir_base = sweep_cfg.get("results_dir_base", "./log/sweep")
    
    def train_sweep():
        """Function called by wandb agent for each run"""
        # Initialize wandb run (required for sweep)
        run = wandb.init(project=project, entity=entity)
        
        # Get parameters from wandb.config
        base_lr = wandb.config.get("base_lr")
        weight_decay = wandb.config.get("weight_decay")
        warmup_factor = wandb.config.get("warmup_factor")
        warmup_iters = wandb.config.get("warmup_iters")
        margin = wandb.config.get("margin")
        ims_per_batch = wandb.config.get("ims_per_batch")
        max_epochs = wandb.config.get("max_epochs")
        
        # Create unique directory for this run
        run_name = run.name.replace(" ", "_") if run.name else run.id
        run_results_dir = os.path.join(results_dir_base, "sweep", sweep_id, run_name)
        
        # Build training command
        cmd_parts = [
            sys.executable,
            "tools/train.py",
            f"--config_file={args.config_file}",
        ]
        
        # Add OUTPUT_DIR (key and value must be separate for YACS)
        cmd_parts.append("OUTPUT_DIR")
        cmd_parts.append(run_results_dir)
        
        # Add hyperparameter overrides (key and value must be separate)
        if base_lr is not None:
            cmd_parts.append("SOLVER.BASE_LR")
            cmd_parts.append(str(base_lr))
        if weight_decay is not None:
            cmd_parts.append("SOLVER.WEIGHT_DECAY")
            cmd_parts.append(str(weight_decay))
        if warmup_factor is not None:
            cmd_parts.append("SOLVER.WARMUP_FACTOR")
            cmd_parts.append(str(warmup_factor))
        if warmup_iters is not None:
            cmd_parts.append("SOLVER.WARMUP_ITERS")
            cmd_parts.append(str(warmup_iters))
        if margin is not None:
            cmd_parts.append("SOLVER.MARGIN")
            cmd_parts.append(str(margin))
        if ims_per_batch is not None:
            cmd_parts.append("SOLVER.IMS_PER_BATCH")
            cmd_parts.append(str(ims_per_batch))
        if max_epochs is not None:
            cmd_parts.append("SOLVER.MAX_EPOCHS")
            cmd_parts.append(str(max_epochs))
        
        # Set environment variables
        env = os.environ.copy()
        # Disable W&B in child process to avoid conflicts
        # Parent process (this function) will handle all W&B logging
        env["WANDB_MODE"] = "disabled"
        
        print(f"\n{'='*80}")
        print(f"Starting sweep run: {run.name}")
        print(f"Command: {shlex.join(cmd_parts)}")
        print(f"{'='*80}\n")
        
        try:
            # Run training and capture output for logging
            import re
            process = subprocess.Popen(
                cmd_parts,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Parse output and log metrics to W&B
            step = 0
            for line in iter(process.stdout.readline, ''):
                print(line, end='')  # Echo output
                
                # Parse training metrics: "Epoch[X] Iteration[Y/Z] Loss: L, Acc: A, Base Lr: R"
                train_match = re.search(r'Epoch\[(\d+)\].*Loss:\s*([\d.]+).*Acc:\s*([\d.]+).*Base Lr:\s*([\d.e-]+)', line)
                if train_match:
                    epoch = int(train_match.group(1))
                    loss = float(train_match.group(2))
                    acc = float(train_match.group(3))
                    lr = float(train_match.group(4))
                    wandb.log({'train/loss': loss, 'train/acc': acc, 'train/lr': lr, 'epoch': epoch}, step=step)
                    step += 1
                
                # Parse validation metrics: "mAP: X%"
                map_match = re.search(r'mAP:\s*([\d.]+)%', line)
                if map_match:
                    mAP = float(map_match.group(1))
                    wandb.log({'val/mAP': mAP}, step=step)
                
                # Parse CMC metrics: "CMC curve, Rank-X :Y%"
                cmc_match = re.search(r'CMC curve, Rank-(\d+)\s*:([\d.]+)%', line)
                if cmc_match:
                    rank = int(cmc_match.group(1))
                    value = float(cmc_match.group(2))
                    wandb.log({f'val/cmc_rank_{rank}': value}, step=step)
            
            process.wait()
            if process.returncode != 0:
                print(f"Training failed with exit code: {process.returncode}")
        except Exception as e:
            print(f"Training failed with error: {e}")
        finally:
            # Finish W&B run
            wandb.finish()
    
    # Determine count
    if args.count is not None:
        count = args.count
    else:
        # Calculate total combinations for grid search
        if method == "grid":
            count = 1
            for param_config in sweep_config["parameters"].values():
                if "values" in param_config:
                    count *= len(param_config["values"])
        else:
            count = sweep_cfg.get("count", 10)
    
    print(f"\n{'='*80}")
    print(f"Sweep Configuration:")
    print(f"  Name: {name}")
    print(f"  Method: {method}")
    print(f"  Metric: {metric['name']} ({metric['goal']})")
    print(f"  Count: {count}")
    print(f"  Parameters: {list(sweep_config['parameters'].keys())}")
    print(f"{'='*80}\n")
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train_sweep, count=count, project=project, entity=entity)
    
    print(f"\n{'='*80}")
    print(f"Sweep completed!")
    print(f"View results at: https://wandb.ai/{entity if entity else 'YOUR_USERNAME'}/{project}/sweeps/{sweep_id}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

