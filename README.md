# battery_transformer

Internal repository for battery and fuel cell transformer modeling and visualization.

## Main scripts
- `battery_transformer.py`, `battery_transformer_HDMR.py`: Transformer models for battery data
- `battery_lstm.py`, `battery_cnn.py`, `battery_tcn.py`, `battery_xgb.py`: Alternative models for battery prediction
- `fc_transformer.py`: Fuel cell transformer model
- `monitor_model.py`: Model monitoring
- `capacity_extract.py`: Data extraction
- `visualize_hw_usage.py`, `visualize_metrics.py`, `battery_drop_visualization.ipynb`: Visualization utilities and notebooks
- Notebooks: Model training and results exploration

Input data expected in the `data/` directory. Results output to `outputs/`.