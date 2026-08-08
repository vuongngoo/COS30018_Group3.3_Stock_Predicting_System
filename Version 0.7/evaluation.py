import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score
)

import torch
import torch.nn as nn

# Configurations
OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model Architecture
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


class DirectionClassifier(nn.Module):
    def __init__(self, model_type: str, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super(DirectionClassifier, self).__init__()
        self.model_type = model_type.lower()
        rnn_map = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}
        
        self.rnn = rnn_map[self.model_type](
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
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        context = self.attention(out)
        return self.fc(context).squeeze(-1)


# Execution Routine
def evaluate_all_12_models():
    print("=== Evaluating All 12 Models (6 Baseline vs 6 Sentiment Enriched) ===")

    # 1. Load Datasets
    sentiment_npz = os.path.join("data_cache", "preprocessed_tensors_sentiment.npz")
    baseline_npz = os.path.join("data_cache", "preprocessed_tensors.npz")

    if not os.path.exists(sentiment_npz):
        raise FileNotFoundError("Missing sentiment tensors file! Run data_preprocess_sentiment.py first.")

    data_sent = np.load(sentiment_npz)
    X_test_sent = data_sent['X_test']
    y_test_sent = data_sent['y_test'].astype(int)

    # Load baseline tensors or isolate technical features subset
    if os.path.exists(baseline_npz):
        data_base = np.load(baseline_npz)
        X_test_base = data_base['X_test']
        y_test_base = data_base['y_test'].astype(int)
    else:
        # Fallback: slice off the last 3 sentiment features
        X_test_base = X_test_sent[:, :, :-3]
        y_test_base = y_test_sent

    configs = [
        {"name": "config_1_standard", "hidden_size": 32, "num_layers": 1, "dropout": 0.2, "short": "cfg1"},
        {"name": "config_2_deep", "hidden_size": 64, "num_layers": 2, "dropout": 0.3, "short": "cfg2"}
    ]
    models = ['lstm', 'gru', 'rnn']

    experiments = [
        {"type": "Baseline (Technical Only)", "folder": "experiments", "X_test": X_test_base, "y_test": y_test_base, "cmap": "Blues"},
        {"type": "Sentiment Enriched", "folder": "experiments_sentiment", "X_test": X_test_sent, "y_test": y_test_sent, "cmap": "Oranges"}
    ]

    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    summary_rows = []

    for row_idx, exp in enumerate(experiments):
        col_count = 0
        for cfg in configs:
            for m_type in models:
                ax = axes[row_idx, col_count]
                model_path = os.path.join(exp["folder"], cfg["name"], f"best_{m_type}_model.pth")
                model_label = f"{exp['type']}\n{cfg['short'].upper()} - {m_type.upper()}"

                if not os.path.exists(model_path):
                    print(f"[Warning] Missing file: {model_path}")
                    ax.text(0.5, 0.5, "File Missing", ha='center', va='center')
                    col_count += 1
                    continue

                X_test = exp["X_test"]
                y_test = exp["y_test"]
                num_features = X_test.shape[2]

                # Model Initialization & Loading
                model = DirectionClassifier(
                    model_type=m_type,
                    input_size=num_features,
                    hidden_size=cfg["hidden_size"],
                    num_layers=cfg["num_layers"],
                    dropout=cfg["dropout"]
                ).to(DEVICE)

                model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
                model.eval()

                # Inference
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(X_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy()

                preds = (probs >= 0.50).astype(int)
                cm = confusion_matrix(y_test, preds)
                tn, fp, fn, tp = cm.ravel()

                # Plot Heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap=exp["cmap"], ax=ax, cbar=False,
                            xticklabels=['Pred 0', 'Pred 1'],
                            yticklabels=['Act 0', 'Act 1'])
                ax.set_title(model_label, fontsize=10, fontweight='bold')

                # Log Metrics
                summary_rows.append({
                    "Experiment": exp["type"],
                    "Config": cfg["name"],
                    "Model": m_type.upper(),
                    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                    "Accuracy (%)": round(accuracy_score(y_test, preds) * 100, 2),
                    "Precision (%)": round(precision_score(y_test, preds, zero_division=0) * 100, 2),
                    "Recall (%)": round(recall_score(y_test, preds, zero_division=0) * 100, 2),
                    "F1 Score (%)": round(f1_score(y_test, preds, zero_division=0) * 100, 2),
                    "ROC AUC (%)": round(roc_auc_score(y_test, probs) * 100, 2)
                })

                col_count += 1

    plt.suptitle("Confusion Matrix Comparison Across All 12 Models\n(Top Row: Technical Baseline | Bottom Row: Sentiment Enriched)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    grid_path = os.path.join(OUTPUT_DIR, "confusion_matrices_all_12_models.png")
    plt.savefig(grid_path, dpi=300)
    plt.close()

    # Save CSV Summary
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "all_12_models_evaluation_summary.csv")
    df_summary.to_csv(csv_path, index=False)

    print(f"\nSaved 12-Grid Confusion Matrix Chart to: '{grid_path}'")
    print(f"Saved Complete CSV Summary to:            '{csv_path}'\n")

    # Display Pretty Table in Terminal
    print("=" * 105)
    print("                                COMPLETE 12-MODEL EVALUATION SUMMARY")
    print("=" * 105)
    print(f"{'Experiment':<25} | {'Config':<17} | {'Model':<5} | {'Acc (%)':<7} | {'Prec (%)':<8} | {'Rec (%)':<7} | {'F1 (%)':<7} | {'AUC (%)':<7}")
    print("-" * 105)
    for row in summary_rows:
        print(f"{row['Experiment']:<25} | {row['Config']:<17} | {row['Model']:<5} | {row['Accuracy (%)']:<7.2f} | {row['Precision (%)']:<8.2f} | {row['Recall (%)']:<7.2f} | {row['F1 Score (%)']:<7.2f} | {row['ROC AUC (%)']:<7.2f}")
    print("=" * 105)


if __name__ == "__main__":
    evaluate_all_12_models()