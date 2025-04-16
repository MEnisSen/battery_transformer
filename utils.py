import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import subprocess, time, psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# ------------------------------
# COMMON
# ------------------------------

def truncated_normal_(tensor, mean=0.0, std=0.2, a=-2.0, b=2.0):
    """
    Custom truncated normal initialization.
    """
    lower, upper = (a - mean) / std, (b - mean) / std
    tensor.data = torch.distributions.Normal(mean, std).rsample(tensor.shape)
    tensor.data = torch.clip(tensor.data, min=a, max=b)
    return tensor

# ------------------------------
# TRANSFORMER Model
# ------------------------------

class InputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_sizes=[4, 3], strides=[2, 2]):
        super(InputEmbedding, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm1d(embed_dim)

        self.cls_token = nn.Parameter(truncated_normal_(torch.empty(1, 1, embed_dim)))
        self.pos_embedding = nn.Parameter(truncated_normal_(torch.empty(1, embed_dim + 1, embed_dim)))

    def forward(self, x):
        """
        x: (batch_size, channels, time_steps)
        Output: (batch_size, seq_len+1, embed_dim)
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = x.permute(0, 2, 1)  # Reshape for transformer (batch_size, seq_len, embed_dim)

        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[:, :x.shape[1], :]

        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        return self.out_proj(attn_output)

class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim, ffn_dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.gelu(self.fc1(x))))

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=16, ffn_dim=1024, drop_path_rate=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.attn(self.norm1(x), mask))
        x = x + self.drop_path2(self.ffn(self.norm2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ffn_dim, drop_path_rate)
            for _ in range(num_blocks)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class SOHTransformer(nn.Module):
    def __init__(self, input_dim=6, embed_dim=256, num_blocks=4, num_heads=16, ffn_dim=1024, drop_path_rate=0.1):
        super(SOHTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ffn_dim, dropout=drop_path_rate, batch_first=True),
            num_layers=num_blocks
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 30)  # Predicting next 100 available_capacity values
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.mlp_head(x[:, -1])

# ------------------------------
# LSTM Model
# ------------------------------

class SOHLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, dropout=0.1):
        super(SOHLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 100)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        return self.fc(x)

# ------------------------------
# CNN Model
# ------------------------------

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_sizes=[4, 3], strides=[2, 2]):
        super(CNNFeatureExtractor, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_sizes[0], stride=strides[0])
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_sizes[1], stride=strides[1])
        self.bn2 = nn.BatchNorm1d(embed_dim)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        x: (batch_size, channels, time_steps)
        """
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.global_pooling(x).squeeze(-1)
        return x
    
class SOHCNN(nn.Module):
    def __init__(self, input_dim=5, embed_dim=256):
        super(SOHCNN, self).__init__()
        self.feature_extractor = CNNFeatureExtractor(input_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 100)  # Regression output
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.mlp_head(x)

# ------------------------------
# TCN Model
# ------------------------------

class TCNBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(output_dim, output_dim, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.residual = nn.Conv1d(input_dim, output_dim, 1) if input_dim != output_dim else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Ensure the same sequence length for residual and output
        min_length = min(x.shape[2], residual.shape[2])
        x = x[:, :, :min_length]
        residual = residual[:, :, :min_length]
        
        return x + residual

class TCNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, embed_dim=256, kernel_size=3, num_layers=3):
        super(TCNFeatureExtractor, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(TCNBlock(input_dim if i == 0 else embed_dim, embed_dim, kernel_size, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = self.global_pooling(x).squeeze(-1)
        return x

class SOHTCN(nn.Module):
    def __init__(self, input_dim=5, embed_dim=256):
        super(SOHTCN, self).__init__()
        self.feature_extractor = TCNFeatureExtractor(input_dim, embed_dim)
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 30)  # Regression output
        )
    
    def forward(self, x):
        x = self.feature_extractor(x)
        return self.mlp_head(x)

# ------------------------------
# DATA PREPROC
# ------------------------------

class SOHDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SOHDatasetLSTM(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def create_sequences(X, y, seq_len):

    sequences = []
    targets = []
    
    for i in range(len(X) - seq_len * 2):
        sequences.append(X[i:i+seq_len])
        targets.append(y[i+seq_len:i+seq_len*2])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def load_and_proc_data(file_list,
                       features=['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc', 'available_capacity (Ah)'],
                       targets = ['available_capacity (Ah)'],
                       SEQ_LEN=100, 
                       BATCH_SIZE=32,
                       model_type=None):
    
    X_seq = []
    y_seq = []

    for file in file_list[:1]:
        df = pd.read_csv(file)
        
        X = df[features].values
        y = df[targets[0]].values

        scaler_data = StandardScaler()
        X = scaler_data.fit_transform(X)
        y = y / 135

        X_seq_temp, y_seq_temp = create_sequences(X, y, SEQ_LEN)
        X_seq.extend(X_seq_temp)
        y_seq.extend(y_seq_temp)

    if not X_seq or not y_seq:
        raise ValueError("No data processed. Please check the file list and ensure it contains valid data.")

    train_size = int(0.8 * len(X_seq))
    val_size = int(0.1 * len(X_seq))
    test_size = len(X_seq) - train_size - val_size

    X_train, y_train = X_seq[:train_size], y_seq[:train_size]
    X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
    X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]

    if model_type == 'lstm':
        train_dataset = SOHDatasetLSTM(X_train, y_train)
        val_dataset = SOHDatasetLSTM(X_val, y_val)
        test_dataset = SOHDatasetLSTM(X_test, y_test)
    else:
        train_dataset = SOHDataset(X_train, y_train)
        val_dataset = SOHDataset(X_val, y_val)
        test_dataset = SOHDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return X, y, train_loader, val_loader, test_loader, scaler_data

def load_and_proc_data_xgb(file_list,
                           features = ['pack_voltage (V)', 'charge_current (A)', 'max_temperature (℃)', 'min_temperature (℃)', 'soc'],
                           targets = ['available_capacity (Ah)']):
    
    
    X_list = []
    y_list = []

    for file in file_list[:5]:
        df = pd.read_csv(file)
        
        X_list.append(df[features].values[:-1])
        
        y_list.append(df[targets[0]].values[1:])

    X = np.vstack(X_list)
    y = np.concatenate(y_list) if y_list else None

    scaler_data = StandardScaler()
    X = scaler_data.fit_transform(X)
    y = y / 135

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_data

# ------------------------------
# MONITORING
# ------------------------------

def monitor_idle_gpu_cpu(duration=10, interval=1):
    
    power_values = []
    gpu_util_values = []
    cpu_util_values = []
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        power, gpu_util = map(float, result.stdout.strip().split(", "))
        cpu_util = psutil.cpu_percent(interval=0.1)
        
        power_values.append(power)
        gpu_util_values.append(gpu_util)
        cpu_util_values.append(cpu_util)
        
        time.sleep(interval)
    
    avg_power = sum(power_values) / len(power_values)
    avg_gpu_util = sum(gpu_util_values) / len(gpu_util_values)
    avg_cpu_util = sum(cpu_util_values) / len(cpu_util_values)
    
    return avg_power, avg_gpu_util, avg_cpu_util

# ------------------------------
# TRAINING & TESTING
# ------------------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, model_save_file="models/best_model.pth", device=torch.device("cpu"), num_epochs=20):
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)  # No squeeze(), keeping (batch_size, seq_len)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_file)

def evaluate_model(model, test_loader, model_save_file, output_save_file, plot_model_name='model', plot_fig = True, device=torch.device("cpu")):
    model.load_state_dict(torch.load(model_save_file))
    model.eval()

    first_test_flag = True

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)

            if first_test_flag and plot_fig:
                pred_out = outputs[0].cpu().numpy()
                target_out = batch_y[0].cpu().numpy()
                first_test_flag = False

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute error metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R²: {r2:.4f}")

    with open(output_save_file, "w") as f:
        f.write(f"Test RMSE: {rmse:.4f}\n")
        f.write(f"Test MAE: {mae:.4f}\n")
        f.write(f"Test R²: {r2:.4f}\n")

    if plot_fig:
        plt.figure(figsize=(10, 5))
        plt.plot(pred_out, label="predicted", linestyle="dashed", alpha=0.7)
        plt.plot(target_out, label="target", linestyle="solid", alpha=0.7)
        
        plt.ylabel("SOH (%)")
        plt.xticks([])
        plt.title(f'{plot_model_name} example')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'outputs/figures/{plot_model_name}_example.png')
