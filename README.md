# battery_transformer

Battery and fuel cell state-of-health and remaining useful life modeling via Transformers, LSTMs, CNNs, TCNs, and XGBoost with visualizations.

## Main scripts
- `battery_transformer.py`, `battery_transformer_HDMR.py`: Transformer models for battery data
- `battery_lstm.py`, `battery_cnn.py`, `battery_tcn.py`, `battery_xgb.py`: Alternative models for battery prediction
- `fc_transformer.py`: Fuel cell transformer model
- `monitor_model.py`: Model monitoring
- `capacity_extract.py`: Data extraction
- `visualize_hw_usage.py`, `visualize_metrics.py`, `battery_drop_visualization.ipynb`: Visualization utilities and notebooks
- Notebooks: Model training and results exploration

Input data expected in the `data/` directory. Results output to `outputs/`.
