# linear_regression_pytorch.py
# Requirements: torch, numpy, scikit-learn, matplotlib, seaborn, pandas
# Example: python linear_regression_pytorch.py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dotenv import load_dotenv

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from prometheus_client import Gauge, Counter, start_http_server
try:
    import psutil  # optional CPU metrics
except Exception:
    psutil = None

# --------------------
# Observability: Prometheus & Galileo
# --------------------
load_dotenv()  # load .env file so GALILEO_API_KEY and others are available
start_http_server(8000)  # expose /metrics on port 8000

# Prometheus metrics
METRIC_EPOCH = Gauge("training_epoch", "Current training epoch")
METRIC_TRAIN_LOSS = Gauge("training_loss", "Average training loss (scaled)")
METRIC_BATCH_COUNT = Counter("training_batch_count", "Total training batches processed")
METRIC_TEST_MSE = Gauge("test_mse", "Test MSE")
METRIC_TEST_R2 = Gauge("test_r2", "Test R2 score")
METRIC_CPU = Gauge("cpu_percent", "CPU usage percent")
METRIC_GPU_MEM_ALLOC = Gauge("gpu_memory_allocated_bytes", "CUDA memory allocated (bytes)")
METRIC_GPU_MEM_RES = Gauge("gpu_memory_reserved_bytes", "CUDA memory reserved (bytes)")

# Galileo setup (optional)
GALILEO_RUN = None
def setup_galileo():
    api_key = os.getenv("GALILEO_API_KEY")
    project_name = os.getenv("GALILEO_PROJECT", "LinearRegression-Observability")
    run_name = os.getenv("GALILEO_RUN_NAME", f"run-{os.getenv('COMPUTERNAME','win')}-{os.getpid()}")
    if not api_key:
        return None
    try:
        import galileo as ga
        run = ga.init(api_key=api_key, project=project_name, run_name=run_name)
        run.log_params({
            "lr": float(LR),
            "batch_size": int(BATCH_SIZE),
            "epochs": int(EPOCHS),
            "weight_decay": float(WEIGHT_DECAY),
            "synthetic": bool(SYNTHETIC),
            "features": int(N_FEATURES),
            "test_size": float(TEST_SIZE)
        })
        return run
    except Exception as e:
        print(f"Galileo logging disabled: {e}")
        return None

GALILEO_RUN = setup_galileo()

# --------------------
# Config / Hyperparams
# --------------------
RANDOM_SEED = 42
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Data options
LOAD_CSV_PATH = os.path.join(os.path.dirname(__file__), "Dataset", "insurance.csv")
SYNTHETIC = False       # set True to use synthetic regression data
N_SAMPLES = 1000
N_FEATURES = 1
NOISE = 15.0
TEST_SIZE = 0.2

# Training hyperparameters
LR = 1e-2
BATCH_SIZE = 64
EPOCHS = 200
WEIGHT_DECAY = 0.0
PRINT_EVERY = 20

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --------------------
# Data Preparation
# --------------------
if not SYNTHETIC and os.path.exists(LOAD_CSV_PATH):
    print(f"✅ Loading dataset from {LOAD_CSV_PATH}")
    df = pd.read_csv(LOAD_CSV_PATH)

    # Auto-detect target column (supports 'charges', 'expenses', 'cost', 'price')
    possible_targets = ["charges", "expenses", "cost", "price"]
    target_col = None
    for col in df.columns:
        col_clean = col.strip().lower()
        if col_clean in possible_targets:
            target_col = col
            break

    if not target_col:
        raise ValueError(
            f"❌ Target column not found. Available columns: {df.columns.tolist()}"
        )

    print(f"🎯 Target column detected: '{target_col}'")

    # Separate features and target
    y = df[target_col].values.reshape(-1, 1)
    X = df.drop(columns=[target_col])

    # Auto-detect categorical and numeric columns
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    print(f"🧠 Detected categorical columns: {categorical_cols}")
    print(f"🔢 Detected numeric columns: {numeric_cols}")

    # One-hot encode categorical variables (if any)
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, drop="first")
        X_encoded = encoder.fit_transform(X[categorical_cols])
        X = np.hstack([X[numeric_cols].values, X_encoded])
    else:
        X = X[numeric_cols].values

else:
    print("⚙️ Using synthetic regression dataset...")
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=NOISE,
        random_state=RANDOM_SEED,
    )

# Ensure y is 2D for scaler
y = y.reshape(-1, 1)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# Standard scaling
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# Convert to tensors
tensor_X_train = torch.tensor(X_train_scaled, dtype=torch.float32)
tensor_y_train = torch.tensor(y_train_scaled, dtype=torch.float32)
tensor_X_test = torch.tensor(X_test_scaled, dtype=torch.float32)
tensor_y_test = torch.tensor(y_test_scaled, dtype=torch.float32)

# DataLoader
train_ds = TensorDataset(tensor_X_train, tensor_y_train)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# --------------------
# Model Definition
# --------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel(in_features=X_train.shape[1]).to(DEVICE)

# --------------------
# Loss & Optimizer
# --------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# --------------------
# Training Loop
# --------------------
train_losses = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        loss = criterion(preds, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        METRIC_BATCH_COUNT.inc()

    avg_loss = float(np.mean(epoch_losses))
    train_losses.append(avg_loss)
    METRIC_EPOCH.set(epoch)
    METRIC_TRAIN_LOSS.set(avg_loss)
    if psutil:
        try:
            METRIC_CPU.set(psutil.cpu_percent(interval=None))
        except Exception:
            pass
    if USE_CUDA:
        try:
            METRIC_GPU_MEM_ALLOC.set(float(torch.cuda.memory_allocated(0)))
            METRIC_GPU_MEM_RES.set(float(torch.cuda.memory_reserved(0)))
        except Exception:
            pass

    if GALILEO_RUN:
        try:
            GALILEO_RUN.log_metrics({"epoch": int(epoch), "train_loss": float(avg_loss)})
        except Exception as e:
            print(f"Galileo metrics log failed: {e}")

    if epoch % PRINT_EVERY == 0 or epoch == 1 or epoch == EPOCHS:
        print(f"Epoch {epoch:03d}/{EPOCHS} - Train MSE (scaled): {avg_loss:.6f}")

# --------------------
# Evaluation
# --------------------
model.eval()
with torch.no_grad():
    preds_test_scaled = model(tensor_X_test.to(DEVICE)).cpu().numpy()

# Inverse transform
preds_test = y_scaler.inverse_transform(preds_test_scaled)
y_test_orig = y_scaler.inverse_transform(tensor_y_test.numpy())

# Metrics
mse = mean_squared_error(y_test_orig, preds_test)
r2 = r2_score(y_test_orig, preds_test)
print(f"\n📊 Test MSE: {mse:.4f}   |   R² Score: {r2:.4f}")
METRIC_TEST_MSE.set(float(mse))
METRIC_TEST_R2.set(float(r2))

if GALILEO_RUN:
    try:
        GALILEO_RUN.log_metrics({"test_mse": float(mse), "test_r2": float(r2)})
        sample_n = min(50, len(X_test_scaled))
        inputs = X_test_scaled[:sample_n].tolist()
        outputs = preds_test[:sample_n].ravel().tolist()
        references = y_test_orig[:sample_n].ravel().tolist()
        GALILEO_RUN.log_predictions(inputs=inputs, outputs=outputs, references=references)
    except Exception as e:
        print(f"Galileo prediction log failed: {e}")

# --------------------
# Visualizations
# --------------------
sns.set(style="whitegrid", context="talk")

# 1️⃣ Loss curve
plt.figure(figsize=(6, 4))
plt.plot(range(1, EPOCHS + 1), train_losses, label="Train MSE (scaled)")
plt.xlabel("Epoch")
plt.ylabel("MSE (scaled)")
plt.title("Training Loss Curve")
plt.legend()
plt.tight_layout()
plt.show()

# 2️⃣ Predicted vs Actual scatter
plt.figure(figsize=(6, 6))
plt.scatter(y_test_orig, preds_test, alpha=0.6)
minv, maxv = min(y_test_orig.min(), preds_test.min()), max(y_test_orig.max(), preds_test.max())
plt.plot([minv, maxv], [minv, maxv], "r--", label="Perfect Fit")
plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Predicted vs Actual (Test set)")
plt.legend()
plt.tight_layout()
plt.show()

# 3️⃣ Residuals distribution
residuals = (y_test_orig - preds_test).ravel()
plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True)
plt.xlabel("Residual (Actual - Predicted)")
plt.title("Residuals Distribution")
plt.tight_layout()
plt.show()
