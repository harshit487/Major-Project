"""
Stage 5: Model Evaluation & Publication-Grade Visualization on the Untouched
HAM10000 Test Split.

Calculates and saves:
- Accuracy, Precision, Recall (Malignant Sensitivity), Specificity (TN / (TN + FP)),
  F1-score, ROC-AUC, and PR-AUC.
- Confusion Matrix (TN, FP, FN, TP) and full scikit-learn classification report.
- Sample counts: Benign test samples, Malignant test samples, Correctly classified,
  Incorrectly classified, and False Negative (missed malignant) analysis.
- Comparison between default threshold (0.50) and validation-selected threshold.
- All 6 required plots saved to `results/figures/`:
  1. training_val_accuracy.png
  2. training_val_loss.png
  3. confusion_matrix.png
  4. roc_curve.png
  5. precision_recall_curve.png
  6. class_distribution.png
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure repository root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_DECISION_THRESHOLD,
    EVALUATION_METRICS_PATH,
    FIGURES_DIR,
    HISTORY_DIR,
    METRICS_DIR,
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
    get_active_decision_threshold,
)
from backend.services.preprocessing import build_tf_dataset


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Calculate all required clinical binary classification metrics on a dataset
    split at a documented decision threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    total_samples = int(len(y_true_arr))
    benign_samples = int(np.sum(y_true_arr == 0))
    malignant_samples = int(np.sum(y_true_arr == 1))
    correct_samples = int(tn + tp)
    incorrect_samples = int(fp + fn)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    specificity = float(tn / max(1, (tn + fp)))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        roc_auc_val = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        roc_auc_val = 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc_val = float(auc(rec_curve, prec_curve))

    cls_report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    cls_report_text = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        digits=4,
        zero_division=0,
    )

    return {
        "decision_threshold": round(float(threshold), 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc_val, 4),
        "pr_auc": round(pr_auc_val, 4),
        "confusion_matrix": {
            "matrix": [[tn, fp], [fn, tp]],
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
        "sample_counts": {
            "total_test_samples": total_samples,
            "benign_test_samples": benign_samples,
            "malignant_test_samples": malignant_samples,
            "correctly_classified_samples": correct_samples,
            "incorrectly_classified_samples": incorrect_samples,
            "false_negatives_missed_malignant": fn,
            "false_positives_benign_flagged": fp,
        },
        "classification_report": cls_report_dict,
        "classification_report_text": cls_report_text,
    }


def generate_evaluation_figures(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    history_dict: Optional[Dict[str, list]] = None,
    figures_dir: Path = FIGURES_DIR,
) -> Dict[str, str]:
    """
    Generate and save the required evaluation figures to `results/figures/`:
    1. training_val_accuracy.png
    2. training_val_loss.png
    3. confusion_matrix.png
    4. roc_curve.png
    5. precision_recall_curve.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_figures: Dict[str, str] = {}

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    y_prob_arr = np.asarray(y_prob, dtype=float).ravel()
    y_pred_arr = (y_prob_arr >= threshold).astype(int)

    # 1 & 2. Training vs Validation Accuracy and Loss graphs
    if history_dict and "accuracy" in history_dict and "loss" in history_dict:
        epochs = list(range(1, len(history_dict["accuracy"]) + 1))

        # Accuracy plot
        acc_path = figures_dir / "training_val_accuracy.png"
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=160)
        ax.plot(epochs, history_dict["accuracy"], "o-", label="Training Accuracy", color="#0d9488", linewidth=2)
        if "val_accuracy" in history_dict:
            ax.plot(epochs, history_dict["val_accuracy"], "s--", label="Validation Accuracy", color="#2563eb", linewidth=2)
        ax.set_title("Training vs. Validation Accuracy (EfficientNetB0)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        fig.savefig(acc_path, bbox_inches="tight")
        plt.close(fig)
        saved_figures["training_val_accuracy"] = str(acc_path.relative_to(PROJECT_ROOT))

        # Loss plot
        loss_path = figures_dir / "training_val_loss.png"
        fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=160)
        ax.plot(epochs, history_dict["loss"], "o-", label="Training Loss", color="#dc2626", linewidth=2)
        if "val_loss" in history_dict:
            ax.plot(epochs, history_dict["val_loss"], "s--", label="Validation Loss", color="#d97706", linewidth=2)
        ax.set_title("Training vs. Validation Binary Cross-Entropy Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        fig.savefig(loss_path, bbox_inches="tight")
        plt.close(fig)
        saved_figures["training_val_loss"] = str(loss_path.relative_to(PROJECT_ROOT))

    # 3. Confusion Matrix Heatmap
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    cm_path = figures_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Benign (0)", "Predicted Malignant (1)"],
        yticklabels=["Actual Benign (0)", "Actual Malignant (1)"],
        cbar=True,
        annot_kws={"size": 13, "weight": "bold"},
        ax=ax,
    )
    ax.set_title(
        f"Test Set Confusion Matrix (Threshold = {threshold:.2f})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Ground-Truth Label", fontsize=10)
    ax.set_xlabel("Model Prediction", fontsize=10)
    plt.tight_layout()
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    saved_figures["confusion_matrix"] = str(cm_path.relative_to(PROJECT_ROOT))

    # 4. ROC Curve
    fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
    roc_auc_val = auc(fpr, tpr)
    roc_path = figures_dir / "roc_curve.png"
    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=160)
    ax.plot(fpr, tpr, color="#0d9488", linewidth=2.5, label=f"EfficientNetB0 (ROC-AUC = {roc_auc_val:.4f})")
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1.2, label="Random Classifier (AUC = 0.50)")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve — Test Split", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Malignant Recall / Sensitivity)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    fig.savefig(roc_path, bbox_inches="tight")
    plt.close(fig)
    saved_figures["roc_curve"] = str(roc_path.relative_to(PROJECT_ROOT))

    # 5. Precision-Recall Curve
    prec_c, rec_c, _ = precision_recall_curve(y_true_arr, y_prob_arr)
    pr_auc_val = auc(rec_c, prec_c)
    pr_path = figures_dir / "precision_recall_curve.png"
    fig, ax = plt.subplots(figsize=(6.8, 5.5), dpi=160)
    ax.plot(rec_c, prec_c, color="#dc2626", linewidth=2.5, label=f"EfficientNetB0 (PR-AUC = {pr_auc_val:.4f})")
    baseline = float(np.mean(y_true_arr == 1))
    ax.axhline(baseline, color="#64748b", linestyle="--", linewidth=1.2, label=f"Malignant Prevalence ({baseline:.2f})")
    ax.set_title("Precision-Recall Curve (Malignant Class) — Test Split", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Malignant Sensitivity)", fontsize=10)
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", frameon=True)
    plt.tight_layout()
    fig.savefig(pr_path, bbox_inches="tight")
    plt.close(fig)
    saved_figures["precision_recall_curve"] = str(pr_path.relative_to(PROJECT_ROOT))

    # 6. Include class_distribution.png reference if already created
    class_dist_path = figures_dir / "class_distribution.png"
    if class_dist_path.exists():
        saved_figures["class_distribution"] = str(class_dist_path.relative_to(PROJECT_ROOT))

    return saved_figures


def run_evaluation(
    processed_dir: Path = DATA_PROCESSED_DIR,
    model_path: Path = MODEL_WEIGHTS_PATH,
    batch_size: int = 32,
    threshold_override: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate the trained model strictly on the untouched `test_split.csv` and
    save all metrics, curves, and plots to `results/`.
    """
    import tensorflow as tf

    test_csv = processed_dir / "test_split.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Test split CSV not found at {test_csv}. "
            "Run `.venv/bin/python scripts/prepare_dataset.py` first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model file not found at {model_path}. "
            "Run `.venv/bin/python scripts/train_model.py` first."
        )

    test_df = pd.read_csv(test_csv)
    if "split" in test_df.columns and not (test_df["split"] == "test").all():
        raise ValueError("Evaluation aborted: test_split.csv contains non-test split records!")

    print(f"[Evaluation] Loading trained model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    active_threshold = (
        float(threshold_override)
        if threshold_override is not None
        else get_active_decision_threshold()
    )

    print(f"[Evaluation] Running inference on {len(test_df):,} untouched test images...")
    test_ds = build_tf_dataset(
        image_paths=test_df["image_path"].tolist(),
        labels=test_df["binary_label"].astype(int).tolist(),
        batch_size=batch_size,
        is_training=False,
    )

    y_prob = model.predict(test_ds, verbose=1).ravel()
    y_true = test_df["binary_label"].astype(int).to_numpy()

    primary_metrics = compute_binary_metrics_at_threshold(y_true, y_prob, active_threshold)
    default_050_metrics = compute_binary_metrics_at_threshold(
        y_true, y_prob, DEFAULT_DECISION_THRESHOLD
    )

    # Load training history if available
    history_json_path = HISTORY_DIR / "training_history.json"
    history_dict = None
    if history_json_path.exists():
        with open(history_json_path, "r", encoding="utf-8") as f:
            history_dict = json.load(f)

    saved_figures = generate_evaluation_figures(
        y_true=y_true,
        y_prob=y_prob,
        threshold=active_threshold,
        history_dict=history_dict,
    )

    # Subsample ROC and PR curve points (~60 points) for interactive Recharts visualization
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_step = max(1, len(fpr) // 60)
    roc_points = [
        {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
        for i in range(0, len(fpr), roc_step)
    ]
    if roc_points[-1]["fpr"] != 1.0:
        roc_points.append({"fpr": 1.0, "tpr": 1.0})

    prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
    pr_step = max(1, len(rec_c) // 60)
    pr_points = [
        {"recall": round(float(rec_c[i]), 4), "precision": round(float(prec_c[i]), 4)}
        for i in range(0, len(rec_c), pr_step)
    ]

    evaluation_payload: Dict[str, Any] = {
        "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "evaluated_split": "test",
        "model_architecture": MODEL_ARCHITECTURE,
        "model_weights_file": model_path.name,
        **primary_metrics,
        "threshold_comparison": {
            "active_threshold_metrics": {
                "threshold": active_threshold,
                "accuracy": primary_metrics["accuracy"],
                "precision": primary_metrics["precision"],
                "recall": primary_metrics["recall"],
                "specificity": primary_metrics["specificity"],
                "f1_score": primary_metrics["f1_score"],
            },
            "default_0_50_threshold_metrics": {
                "threshold": DEFAULT_DECISION_THRESHOLD,
                "accuracy": default_050_metrics["accuracy"],
                "precision": default_050_metrics["precision"],
                "recall": default_050_metrics["recall"],
                "specificity": default_050_metrics["specificity"],
                "f1_score": default_050_metrics["f1_score"],
            },
        },
        "curves": {
            "roc_curve": roc_points,
            "pr_curve": pr_points,
        },
        "training_history": history_dict or {},
        "figures": saved_figures,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_payload, f, indent=2)

    # Save per-sample test predictions for auditing false negatives / false positives
    test_results_df = test_df.copy()
    test_results_df["malignant_probability"] = np.round(y_prob, 6)
    test_results_df["benign_probability"] = np.round(1.0 - y_prob, 6)
    test_results_df["predicted_label"] = (y_prob >= active_threshold).astype(int)
    test_results_df["predicted_class"] = test_results_df["predicted_label"].map({0: "Benign", 1: "Malignant"})
    test_results_df["correct"] = test_results_df["predicted_label"] == test_results_df["binary_label"]
    test_results_df.to_csv(METRICS_DIR / "test_predictions_audit.csv", index=False)

    print("\n================ TEST SET EVALUATION SUMMARY ================")
    print(f"Decision Threshold       : {active_threshold:.4f}")
    print(f"Accuracy                 : {primary_metrics['accuracy']:.4f}")
    print(f"Precision (Malignant)    : {primary_metrics['precision']:.4f}")
    print(f"Recall / Sensitivity     : {primary_metrics['recall']:.4f}")
    print(f"Specificity (Benign)     : {primary_metrics['specificity']:.4f}")
    print(f"F1-Score                 : {primary_metrics['f1_score']:.4f}")
    print(f"ROC-AUC                  : {primary_metrics['roc_auc']:.4f}")
    print(f"PR-AUC                   : {primary_metrics['pr_auc']:.4f}")
    print(f"Benign Test Samples      : {primary_metrics['sample_counts']['benign_test_samples']:,}")
    print(f"Malignant Test Samples   : {primary_metrics['sample_counts']['malignant_test_samples']:,}")
    print(f"Correctly Classified     : {primary_metrics['sample_counts']['correctly_classified_samples']:,}")
    print(f"Incorrectly Classified   : {primary_metrics['sample_counts']['incorrectly_classified_samples']:,}")
    print(f"False Negatives (Missed) : {primary_metrics['sample_counts']['false_negatives_missed_malignant']:,}")
    print("=============================================================\n")
    print(primary_metrics["classification_report_text"])
    print(f"[Complete] Saved evaluation metrics to: {EVALUATION_METRICS_PATH}")

    return evaluation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained HAM10000 binary model on the untouched test split.")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_WEIGHTS_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    try:
        run_evaluation(
            processed_dir=args.processed_dir,
            model_path=args.model_path,
            batch_size=args.batch_size,
            threshold_override=args.threshold,
        )
    except Exception as exc:
        print(f"[Error] Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
