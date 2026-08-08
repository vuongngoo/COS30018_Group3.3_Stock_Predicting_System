import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_hypothesis_and_evidence():
    print("=== Running Hypothesis Testing & Empirical Evidence Generation ===")

    # 1. Load 12-Model Summary Data
    summary_path = os.path.join("all_12_models_evaluation_summary.csv")
    if not os.path.exists(summary_path):
        summary_path = os.path.join(OUTPUT_DIR, "all_12_models_evaluation_summary.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Could not find summary CSV at '{summary_path}'!")

    df = pd.read_csv(summary_path)

    # Separate and sort Baseline vs Sentiment runs to ensure direct pairing
    base = df[df['Experiment'] == 'Baseline (Technical Only)'].sort_values(by=['Config', 'Model']).reset_index(drop=True)
    sent = df[df['Experiment'] == 'Sentiment Enriched'].sort_values(by=['Config', 'Model']).reset_index(drop=True)

    # 2. Hypothesis Testing (Paired t-tests & Wilcoxon Signed-Rank Tests)
    t_stat_acc, p_val_acc = stats.ttest_rel(sent['Accuracy (%)'], base['Accuracy (%)'])
    t_stat_auc, p_val_auc = stats.ttest_rel(sent['ROC AUC (%)'], base['ROC AUC (%)'])

    w_stat_acc, w_p_val_acc = stats.wilcoxon(sent['Accuracy (%)'], base['Accuracy (%)'])
    w_stat_auc, w_p_val_auc = stats.wilcoxon(sent['ROC AUC (%)'], base['ROC AUC (%)'])

    # 3. Compute Specificity & Mode Collapse Flags
    base['Specificity (%)'] = (base['TN'] / (base['TN'] + base['FP']) * 100).round(2)
    sent['Specificity (%)'] = (sent['TN'] / (sent['TN'] + sent['FP']) * 100).round(2)

    evidence_df = pd.DataFrame({
        'Config': base['Config'],
        'Model': base['Model'],
        'Baseline_Acc (%)': base['Accuracy (%)'],
        'Sentiment_Acc (%)': sent['Accuracy (%)'],
        'Acc_Diff (%)': (sent['Accuracy (%)'] - base['Accuracy (%)']).round(2),
        'Baseline_AUC (%)': base['ROC AUC (%)'],
        'Sentiment_AUC (%)': sent['ROC AUC (%)'],
        'AUC_Diff (%)': (sent['ROC AUC (%)'] - base['ROC AUC (%)']).round(2),
        'Baseline_TN': base['TN'],
        'Sentiment_TN': sent['TN'],
        'Baseline_Specificity (%)': base['Specificity (%)'],
        'Sentiment_Specificity (%)': sent['Specificity (%)'],
        'Mode_Collapsed': (sent['TN'] == 0)
    })

    # Save Evidence DataFrame to CSV
    csv_out_path = os.path.join(OUTPUT_DIR, "empirical_evidence_analysis.csv")
    evidence_df.to_csv(csv_out_path, index=False)

    # 4. Save Statistical Test Summary to Text File
    stat_summary_path = os.path.join(OUTPUT_DIR, "hypothesis_testing_report.txt")
    with open(stat_summary_path, "w") as f:
        f.write("=========================================================\n")
        f.write("      STATISTICAL HYPOTHESIS TESTING REPORT\n")
        f.write("=========================================================\n\n")
        f.write("Null Hypothesis (H0): Sentiment features cause NO performance difference.\n")
        f.write("Alt Hypothesis  (H1): Sentiment features cause a significant performance difference.\n\n")
        f.write("--- ACCURACY TEST RESULTS ---\n")
        f.write(f"Baseline Mean Accuracy  : {base['Accuracy (%)'].mean():.2f}%\n")
        f.write(f"Sentiment Mean Accuracy : {sent['Accuracy (%)'].mean():.2f}%\n")
        f.write(f"Paired t-statistic      : {t_stat_acc:.4f}\n")
        f.write(f"p-value (t-test)        : {p_val_acc:.4f}\n")
        f.write(f"Wilcoxon W-statistic    : {w_stat_acc:.4f}\n")
        f.write(f"p-value (Wilcoxon)      : {w_p_val_acc:.4f}\n")
        f.write(f"Conclusion              : {'REJECT H0' if p_val_acc < 0.05 else 'FAIL TO REJECT H0 (Not Significant)'}\n\n")
        f.write("--- ROC AUC TEST RESULTS ---\n")
        f.write(f"Baseline Mean ROC AUC   : {base['ROC AUC (%)'].mean():.2f}%\n")
        f.write(f"Sentiment Mean ROC AUC  : {sent['ROC AUC (%)'].mean():.2f}%\n")
        f.write(f"Paired t-statistic      : {t_stat_auc:.4f}\n")
        f.write(f"p-value (t-test)        : {p_val_auc:.4f}\n")
        f.write(f"Wilcoxon W-statistic    : {w_stat_auc:.4f}\n")
        f.write(f"p-value (Wilcoxon)      : {w_p_val_auc:.4f}\n")
        f.write(f"Conclusion              : {'REJECT H0' if p_val_auc < 0.05 else 'FAIL TO REJECT H0 (Not Significant)'}\n")

    # 5. Generate Evidence Visualization Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(base))
    width = 0.35

    # Plot A: Accuracy Comparison vs Class Ratio
    axes[0].bar(x - width/2, base['Accuracy (%)'], width, label='Baseline (Technical)', color='#1f77b4', alpha=0.85)
    axes[0].bar(x + width/2, sent['Accuracy (%)'], width, label='Sentiment Enriched', color='#ff7f0e', alpha=0.85)
    axes[0].axhline(51.12, color='red', linestyle='--', linewidth=1.5, label='Majority Class Ratio (51.12%)')
    axes[0].set_ylabel('Accuracy (%)', fontweight='bold')
    axes[0].set_title('A: Accuracy Comparison & Class Baseline', fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{c[:4].upper()}-{m}" for c, m in zip(base['Config'], base['Model'])], rotation=45)
    axes[0].legend()
    axes[0].set_ylim(40, 60)

    # Plot B: True Negative (TN) Count Showing Mode Collapse
    axes[1].bar(x - width/2, base['TN'], width, label='Baseline TN Count', color='#2ca02c', alpha=0.85)
    axes[1].bar(x + width/2, sent['TN'], width, label='Sentiment TN Count', color='#d62728', alpha=0.85)
    axes[1].set_ylabel('True Negative (TN) Count', fontweight='bold')
    axes[1].set_title('B: Specificity Loss & Mode Collapse (TN = 0)', fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{c[:4].upper()}-{m}" for c, m in zip(base['Config'], base['Model'])], rotation=45)
    axes[1].legend()

    plt.tight_layout()
    plot_out_path = os.path.join(OUTPUT_DIR, "empirical_evidence_plots.png")
    plt.savefig(plot_out_path, dpi=300)
    plt.close()

    print(f"\n[Success] Created Files in '{OUTPUT_DIR}':")
    print(f"  1. Evidence Analysis CSV  -> '{csv_out_path}'")
    print(f"  2. Hypothesis Report TXT  -> '{stat_summary_path}'")
    print(f"  3. Diagnostic Charts PNG  -> '{plot_out_path}'")


if __name__ == "__main__":
    run_hypothesis_and_evidence()