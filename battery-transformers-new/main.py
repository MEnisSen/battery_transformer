import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
import pandas as pd

from src.data_loader import get_data_loaders, get_data_loaders_xgb
from src.models import lstm, tcn, transformer, cnn, xgboost_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.monitoring import GpuMonitor, monitor_idle_gpu_cpu

# Mapping from model name string to model class
MODEL_MAP = {
    'lstm': lstm.SOHLSTM,
    'tcn': tcn.SOHTCN,
    'transformer': transformer.SOHTransformer,
    'transformer_hdmr': transformer.SOHTransformerHDMR,
    'cnn': cnn.SOHCNN,
    'xgb': xgboost_model.SOHXGBoost  # Wrapper class for XGBoost
}

def load_config(config_path):
    """Loads YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main(config_path):
    """Main function to run experiments based on config."""
    config = load_config(config_path)

    # --- Setup Output Directories ---
    os.makedirs(config['outputs']['logs_dir'], exist_ok=True)
    os.makedirs(config['outputs']['models_dir'], exist_ok=True)
    os.makedirs(config['outputs']['figures_dir'], exist_ok=True)

    # --- Monitoring Setup ---
    monitoring = config.get('monitoring', {}).get('enable', False)
    monitor = None
    if monitoring:
        idle_duration = config['monitoring'].get('idle_duration', 1)
        avg_power, avg_gpu_util, avg_cpu_util = monitor_idle_gpu_cpu(duration=idle_duration)
        print(f"\nAverage idle values over {idle_duration} seconds: \nAVG_GPU_POWER = {avg_power}, AVG_GPU_UTIL = {avg_gpu_util}, AVG_CPU_UTIL = {avg_cpu_util}\n")
        monitor = GpuMonitor(avg_power, avg_gpu_util, avg_cpu_util)

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and config['training'].get('use_gpu', True) else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    data_config = config['data']
    model_type = config['model']['type']
    file_list = [os.path.join(data_config['directory'], f) for f in os.listdir(data_config['directory']) if f.endswith(".csv")]
    print("Data files:", file_list)

    if model_type == 'xgb':
        X_train, X_val, X_test, y_train, y_val, y_test, scaler_data = get_data_loaders_xgb(
            file_list=file_list,
            features=data_config['features'],
            targets=data_config['targets'],
            test_size=data_config.get('test_split_ratio', 0.2), # Example, adjust as needed
            val_size=data_config.get('val_split_ratio', 0.15)   # Example, adjust as needed
        )
        # XGBoost handles data differently, no DataLoader needed for fit/predict
        train_loader, val_loader, test_loader = None, None, None # Placeholder
    else:
        train_loader, val_loader, test_loader, scaler_data = get_data_loaders(
            file_list=file_list,
            features=data_config['features'],
            targets=data_config['targets'],
            seq_len=data_config['seq_len'],
            batch_size=config['training']['batch_size'],
            model_type=model_type, # Pass model_type for potential dataset variations
            train_split_ratio=data_config.get('train_split_ratio', 0.7),
            val_split_ratio=data_config.get('val_split_ratio', 0.15)
        )

    # --- Model Initialization ---
    model_config = config['model']
    model_class = MODEL_MAP.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    # Pass NUM_FEATURES dynamically
    num_features = len(data_config['features'])
    if hasattr(model_class, 'input_dim'): # Check if model expects input_dim
        model_params = {k: v for k, v in model_config['params'].items() if k != 'type'}
        model_params['input_dim'] = num_features
        if model_type != 'xgb':
            model_instance = model_class(**model_params).to(device)
        else:
            model_instance = model_class(**model_params) # XGBoost wrapper doesn't need .to(device)
    else:
        # Handle models that don't take input_dim (like maybe XGB wrapper)
        model_params = {k: v for k, v in model_config['params'].items() if k != 'type'}
        model_instance = model_class(**model_params)

    print(f"Initialized model: {model_type}")
    if model_type != 'xgb':
        print(model_instance)

    # --- Training Setup ---
    training_config = config['training']
    output_config = config['outputs']
    model_save_path = os.path.join(output_config['models_dir'], f"best_{model_type}.pth")
    log_train_path = os.path.join(output_config['logs_dir'], f"log_training_{model_type}.csv")

    if model_type != 'xgb':
        criterion = nn.MSELoss() # Make configurable if needed
        optimizer = optim.AdamW(
            model_instance.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 1e-4)
        )

        trainer = Trainer(
            model=model_instance,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            monitor=monitor,
            log_file=log_train_path,
            monitor_interval=config['monitoring'].get('train_interval', 1)
        )

        # --- Training ---
        print("Starting training...")
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=training_config['num_epochs'],
            model_save_path=model_save_path
        )
        print("Training finished.")

    else: # XGBoost Training
        print("Starting XGBoost training...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        model_instance.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        model_save_path_xgb = os.path.join(output_config['models_dir'], f"best_{model_type}.json")
        model_instance.save_model(model_save_path_xgb)
        model_save_path = model_save_path_xgb # Update save path for evaluation
        print("XGBoost training finished.")

    # --- Evaluation Setup ---
    eval_config = config['evaluation']
    log_test_path = os.path.join(output_config['logs_dir'], f"log_testing_{model_type}.csv")
    results_save_path = os.path.join(output_config['logs_dir'], f"error_results_{model_type}.txt")
    figure_save_path = os.path.join(output_config['figures_dir'], f"{model_type}_example.png")

    evaluator = Evaluator(
        model=model_instance,
        device=device,
        monitor=monitor,
        log_file=log_test_path,
        monitor_interval=config['monitoring'].get('test_interval', 0.01)
    )

    # --- Evaluation ---
    print("Starting evaluation...")
    start_time = time.time()
    if model_type == 'xgb':
        evaluator.evaluate_xgb(
            X_test=X_test,
            y_test=y_test,
            results_save_path=results_save_path,
            # XGBoost evaluator doesn't need model_path or plot name like this
            # Add plotting capabilities if needed
        )
    else:
        evaluator.evaluate(
            test_loader=test_loader,
            model_path=model_save_path,
            results_save_path=results_save_path,
            plot_fig=eval_config.get('plot_figure', True),
            figure_save_path=figure_save_path,
            plot_model_name=model_type # Use model_type for plot title
        )
    eval_time = time.time() - start_time
    print(f"Evaluation finished in {eval_time:.2f} seconds.")

    if monitor:
        monitor.stop()
        print("Monitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Battery SOH Prediction Experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (e.g., config/lstm_config.yaml)")
    args = parser.parse_args()
    main(args.config)
