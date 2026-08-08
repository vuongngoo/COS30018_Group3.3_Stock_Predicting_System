import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DATA_PATH = os.path.join("data_cache", "preprocessed_tensors.npz")
BASE_EXP_DIR = "experiments"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Temporal Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super(TemporalAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        scores = self.attn(rnn_out)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(rnn_out * weights, dim=1)


# Classification Architecture
class DirectionClassifier(nn.Module):
    def __init__(self, model_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(DirectionClassifier, self).__init__()
        self.model_type = model_type.lower()
        
        rnn_map = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
        if self.model_type not in rnn_map:
            raise ValueError(f"Invalid model_type '{model_type}'. Choose from ['lstm', 'gru', 'rnn'].")
            
        rnn_layer = rnn_map[self.model_type]
        
        self.rnn = rnn_layer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.attention = TemporalAttention(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Raw logit output for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        context = self.attention(out)
        return self.fc(context).squeeze(-1)


# Experiment Configurations
CONFIGURATIONS = [
    {
        "name": "config_1_standard",
        "lr": 0.0005,
        "batch_size": 32,
        "epochs": 100,
        "hidden_size": 32,
        "num_layers": 1,
        "dropout": 0.2,
        "patience": 10,
        "desc": "Single layer with Binary Cross Entropy loss"
    },
    {
        "name": "config_2_deep",
        "lr": 0.0003,
        "batch_size": 64,
        "epochs": 100,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.3,
        "patience": 12,
        "desc": "2-layer stacked RNN with dropout 0.3"
    }
]


# Data Loader Function
def load_data_loaders(data_path: str, batch_size: int):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"'{data_path}' not found! Ensure preprocessed_tensors.npz exists.")

    data = np.load(data_path)
    
    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32)
    
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.float32)
    
    X_test = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test = torch.tensor(data['y_test'], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    num_features = X_train.shape[2]
    return train_loader, val_loader, test_loader, num_features


# Single Model Trainer
def train_single_model(model_type: str, cfg: dict, save_dir: str, train_loader, val_loader, test_loader, num_features: int):
    save_path = os.path.join(save_dir, f"best_{model_type}_model.pth")

    model = DirectionClassifier(
        model_type=model_type,
        input_size=num_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(DEVICE)

    # Binary Cross Entropy with Logits Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []

    print(f"\n  [Training {model_type.upper()} Classifier ({num_features} features)]...")
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"    --> Early stopping triggered at epoch {epoch}.")
                break

    # Plot Loss Curve
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train BCE Loss', color='navy', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val BCE Loss', color='crimson', linestyle='--', linewidth=2)
    plt.title(f'Classification Loss Curve: {model_type.upper()} ({cfg["name"]})')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curve_{model_type}.png"), dpi=300)
    plt.close()

    # Test Evaluation
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model.eval()

    test_logits = []
    test_targets = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            logits = model(batch_X)
            test_logits.extend(logits.cpu().numpy())
            test_targets.extend(batch_y.numpy())

    test_probs = torch.sigmoid(torch.tensor(test_logits)).numpy()
    test_preds = (test_probs >= 0.5).astype(int)
    test_targets = np.array(test_targets, dtype=int)

    acc = accuracy_score(test_targets, test_preds) * 100
    prec = precision_score(test_targets, test_preds, zero_division=0) * 100
    rec = recall_score(test_targets, test_preds, zero_division=0) * 100
    f1 = f1_score(test_targets, test_preds, zero_division=0) * 100
    auc = roc_auc_score(test_targets, test_probs) * 100

    print(f"  --> {model_type.upper()} Finished | Acc: {acc:.2f}% | Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | AUC: {auc:.2f}%")

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_Score": f1,
        "ROC_AUC": auc,
        "Saved_Path": save_path
    }


# Main Execution Routine
def main():
    print(f"Using Device: {DEVICE}")
    print("Task: Binary Classification (Predicting Day +1 Up/Down Direction).")

    all_results = {}
    summary_rows = []

    for cfg in CONFIGURATIONS:
        config_folder_name = cfg["name"]
        save_dir = os.path.join(BASE_EXP_DIR, config_folder_name)
        os.makedirs(save_dir, exist_ok=True)

        print("\n" + "=" * 70)
        print(f" CLASSIFICATION CONFIGURATION: {config_folder_name.upper()}")
        print(f" Specs: LR={cfg['lr']}, Batch={cfg['batch_size']}, Hidden={cfg['hidden_size']}, Layers={cfg['num_layers']}")
        print("=" * 70)

        train_loader, val_loader, test_loader, num_features = load_data_loaders(DATA_PATH, cfg["batch_size"])

        config_results = {}
        for model_type in ['lstm', 'gru', 'rnn']:
            metrics = train_single_model(
                model_type=model_type,
                cfg=cfg,
                save_dir=save_dir,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                num_features=num_features
            )
            config_results[model_type.upper()] = metrics

            summary_rows.append({
                "Config": config_folder_name,
                "Model": model_type.upper(),
                "Accuracy (%)": round(metrics["Accuracy"], 2),
                "Precision (%)": round(metrics["Precision"], 2),
                "Recall (%)": round(metrics["Recall"], 2),
                "F1 Score (%)": round(metrics["F1_Score"], 2),
                "ROC AUC (%)": round(metrics["ROC_AUC"], 2)
            })

        all_results[config_folder_name] = config_results

    # Save Metrics Summary CSV
    metrics_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(BASE_EXP_DIR, "classification_metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n Saved classification metrics summary CSV to: '{csv_path}'")

    print("CLASSIFICATION EXPERIMENT SUMMARY")
    print(f"{'Config Folder':<22} | {'Model':<6} | {'Accuracy':<10} | {'Precision':<11} | {'F1 Score':<10} | {'ROC AUC':<10}")

    for cfg_name, models in all_results.items():
        for model_name, metrics in models.items():
            print(f"{cfg_name:<22} | {model_name:<6} | {metrics['Accuracy']:<9.2f}% | {metrics['Precision']:<10.2f}% | {metrics['F1_Score']:<9.2f}% | {metrics['ROC_AUC']:<9.2f}%")



if __name__ == "__main__":
    main()