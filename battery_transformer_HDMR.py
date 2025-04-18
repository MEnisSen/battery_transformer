import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import SOHTransformerHDMR, load_and_proc_data, monitor_idle_gpu_cpu, train_model, evaluate_model, SOHDataset

# MONITORING =============================================================================================

import threading, subprocess, time, psutil

avg_time = 1
avg_power, avg_gpu_util, avg_cpu_util = monitor_idle_gpu_cpu(duration=avg_time)

print(f'\nAverage values over {avg_time} seconds: \nAVG_GPU_POWER = {avg_power}, AVG_GPU_UTIL = {avg_gpu_util}, AVG_CPU_UTIL = {avg_cpu_util}\n')

monitoring = True

def monitor_gpu(log_file = 'gpu_usage_log.csv', interval = 10):

    query_params = [
        "timestamp", "power.draw", "memory.used", "memory.total",
        "utilization.gpu", "utilization.memory", "temperature.gpu",
        "fan.speed", "clocks.sm", "clocks.gr"
    ]
    
    query_str = ",".join(query_params)
    
    with open(log_file, "w") as f:
        f.write("Timestamp,Power (W),Memory Used (MB),Memory Total (MB),GPU Util (%),"
                "Memory Util (%),Temp (C),Fan Speed (%),Clock SM (MHz),Clock Mem (MHz),"
                "CPU Usage (%)\n")
    
    while monitoring:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query_str, "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )

        gpu_data = list(map(float, result.stdout.strip().split(", ")[1:]))
        gpu_data[0] = gpu_data[0] - avg_power
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cpu_usage = psutil.cpu_percent() - avg_cpu_util
        
        log_entry = f"{timestamp}," + ",".join(list(map(str, gpu_data))) + f",{cpu_usage}\n"
        
        with open(log_file, "a") as f:
            f.write(log_entry)

        time.sleep(interval)

# DATA PREPROC ===========================================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
directory = "data/battery_scaled"
file_list = sorted([directory+'/'+f for f in os.listdir(directory) if f.endswith(".csv")])
targets = ['available_capacity (Ah)']

SEQ_LEN = 100
BATCH_SIZE = 32
features = ['soc', 'pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'available_capacity (Ah)']
NUM_FEATURES = len(features)

_, _, train_loader, val_loader, test_loader, scaler_data = load_and_proc_data(file_list,
                                                                              features = features,
                                                                              targets=targets,
                                                                              SEQ_LEN = SEQ_LEN,
                                                                              BATCH_SIZE = BATCH_SIZE)

# Save the underlying tensors of the dataset
X_test_tensor = test_loader.dataset.X
y_test_tensor = test_loader.dataset.y

torch.save({'X': X_test_tensor, 'y': y_test_tensor}, 'test_dataset.pt')

# MODELS - TRAINING ======================================================================================

model = SOHTransformerHDMR(input_dim=NUM_FEATURES, embed_dim=256).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)

monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_training_TRANSFORMER_HDMR_SUNUM.csv', 1), daemon=True)
monitor_thread.start()

train_model(model, train_loader, val_loader, criterion, optimizer, "models/best_TRANSFORMER_HDMR_SUNUM.pth", device, num_epochs=1)

monitoring = False
time.sleep(2)

# MODELS - TESTING =======================================================================================
'''
data = torch.load('test_dataset.pt')
X_loaded = data['X']
y_loaded = data['y']

# Recreate dataset and dataloader
test_dataset = SOHDataset(X_loaded, y_loaded)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
'''
monitoring = True
monitor_thread = threading.Thread(target=monitor_gpu, args=('outputs/log_testing_TRANSFORMER_HDMR_SUNUM.csv', 0.01), daemon=True)
monitor_thread.start()

start_time = time.time()
evaluate_model(model, test_loader, "models/best_TRANSFORMER_HDMR_SUNUM.pth", 'outputs/error_results_TRANSFORMER_HDMR_SUNUM.txt', 'transformer_HDMR_SUNUM', plot_fig = True, device=device)
print(f'{time.time()-start_time} seconds\n')

monitoring = False