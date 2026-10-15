"""
Stage 4: Deep Learning Model Architecture & Transfer Learning Pipeline
(EfficientNetB0 on HAM10000 Binary Classification).

Architecture:
1. Pretrained EfficientNetB0 backbone (include_top=False, input_shape=(224, 224, 3))
2. GlobalAveragePooling2D layer
3. Dropout layer
4. Dense(1, activation='sigmoid') output layer estimating P(Malignant | image)

Training Protocol:
- Reproducible random seed across Python, NumPy, and TensorFlow.
- Phase 1: Train classification head with frozen EfficientNetB0 backbone.
- Phase 2: Fine-tune top N layers of EfficientNetB0 (keeping BatchNormalization
  layers frozen to preserve pretrained running statistics).
- Uses BinaryCrossentropy loss, Adam optimizer, class weights for imbalance,
  EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint on validation ROC-AUC.
- Selects decision threshold using validation data ONLY and saves model_config.json
  plus training history artifacts.
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure repository root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_PROCESSED_DIR,
    DEFAULT_DECISION_THRESHOLD,
    FIGURES_DIR,
    HISTORY_DIR,
    INPUT_SHAPE,
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_PATH,
    MODEL_WEIGHTS_PATH,
    RANDOM_SEED,
)
from backend.services.preprocessing import (
    build_tf_dataset,
    save_augmentation_sample_grid,
)
from scripts.prepare_dataset import compute_class_weights


def set_reproducible_seeds(seed: int = RANDOM_SEED) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    import random
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def build_efficientnetb0_binary_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    dropout_rate: float = 0.35,
    learning_rate: float = 1e-3,
    weights: Optional[str] = "imagenet",
):
    """
    Construct the EfficientNetB0 transfer-learning binary classifier.

    Built as a unified Functional API graph from `base_model.input` to the
    sigmoid output so that convolutional feature maps (`top_conv` / `top_activation`)
    are directly accessible for Grad-CAM explainability.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses, metrics

    try:
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize EfficientNetB0 with weights='{weights}'. "
            "If running offline without cached ImageNet weights, verify internet "
            "connectivity or run with `--weights none` for uninitialized weights "
            f"while preserving the exact EfficientNetB0 architecture. Original error: {exc}"
        ) from exc

    # Freeze all backbone layers initially for Phase 1 head training
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32", name="malignant_prob")(x)

    model = models.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="EfficientNetB0_HAM10000_Binary",
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy", threshold=DEFAULT_DECISION_THRESHOLD),
            metrics.Precision(name="precision", thresholds=DEFAULT_DECISION_THRESHOLD),
            metrics.Recall(name="recall", thresholds=DEFAULT_DECISION_THRESHOLD),
            metrics.AUC(name="auc", curve="ROC"),
            metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def unfreeze_top_backbone_layers(
    model,
    num_layers_to_unfreeze: int = 30,
    fine_tune_lr: float = 1e-4,
) -> int:
    """
    Unfreeze the top `num_layers_to_unfreeze` layers of the EfficientNetB0
    backbone for Phase 2 fine-tuning, while keeping all BatchNormalization
    layers frozen to prevent corrupting pretrained running mean/variance.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, optimizers, losses, metrics

    # Identify backbone layers (all layers prior to global_avg_pool)
    gap_index = [i for i, lyr in enumerate(model.layers) if lyr.name == "global_avg_pool"][0]
    backbone_layers = model.layers[:gap_index]

    for lyr in backbone_layers:
        lyr.trainable = False

    unfrozen_count = 0
    for lyr in backbone_layers[-num_layers_to_unfreeze:]:
        if not isinstance(lyr, layers.BatchNormalization):
            lyr.trainable = True
            unfrozen_count += 1

    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(name="accuracy", threshold=DEFAULT_DECISION_THRESHOLD),
            metrics.Precision(name="precision", thresholds=DEFAULT_DECISION_THRESHOLD),
            metrics.Recall(name="recall", thresholds=DEFAULT_DECISION_THRESHOLD),
            metrics.AUC(name="auc", curve="ROC"),
            metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return unfrozen_count


def select_validation_threshold(
    y_val_true: np.ndarray,
    y_val_prob: np.ndarray,
    min_specificity: float = 0.75,
) -> Dict[str, Any]:
    """
    Evaluate candidate decision thresholds on the VALIDATION set ONLY.
    Computes Youden's J statistic (Sensitivity + Specificity - 1) and F1-score
    across thresholds to find a balanced threshold that prioritizes malignant
    sensitivity while maintaining clinically reasonable specificity.
    """
    from sklearn.metrics import confusion_matrix, f1_score

    y_true = np.asarray(y_val_true, dtype=int).ravel()
    y_prob = np.asarray(y_val_prob, dtype=float).ravel()

    best_threshold = DEFAULT_DECISION_THRESHOLD
    best_score = -1.0
    best_stats: Dict[str, float] = {}

    candidate_thresholds = np.linspace(0.25, 0.75, 51)
    for thresh in candidate_thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(1, (tp + fn))
        specificity = tn / max(1, (tn + fp))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        youden_j = sensitivity + specificity - 1.0

        # Prioritize Youden's J among thresholds meeting minimum specificity
        if specificity >= min_specificity and youden_j > best_score:
            best_score = youden_j
            best_threshold = float(round(thresh, 4))
            best_stats = {
                "val_sensitivity": round(float(sensitivity), 4),
                "val_specificity": round(float(specificity), 4),
                "val_f1": round(float(f1), 4),
                "val_youden_j": round(float(youden_j), 4),
            }

    # Compute default 0.50 stats for comparison
    y_pred_50 = (y_prob >= 0.50).astype(int)
    tn50, fp50, fn50, tp50 = confusion_matrix(y_true, y_pred_50, labels=[0, 1]).ravel()
    default_stats = {
        "threshold": 0.50,
        "val_sensitivity": round(float(tp50 / max(1, tp50 + fn50)), 4),
        "val_specificity": round(float(tn50 / max(1, tn50 + fp50)), 4),
        "val_f1": round(float(f1_score(y_true, y_pred_50, zero_division=0)), 4),
    }

    return {
        "selected_threshold": best_threshold,
        "selection_criterion": f"Maximized Youden's J on validation set (specificity >= {min_specificity})",
        "selected_threshold_val_metrics": best_stats,
        "default_0_50_val_metrics": default_stats,
    }


def merge_histories(hist_head: Dict[str, list], hist_ft: Optional[Dict[str, list]] = None) -> Dict[str, list]:
    """Concatenate Phase 1 (head training) and Phase 2 (fine-tuning) metric histories."""
    merged: Dict[str, list] = {}
    for k, v in hist_head.items():
        merged[k] = [float(x) for x in v]
    if hist_ft:
        for k, v in hist_ft.items():
            if k in merged:
                merged[k].extend([float(x) for x in v])
            else:
                merged[k] = [float(x) for x in v]
    return merged


def run_training(
    processed_dir: Path = DATA_PROCESSED_DIR,
    model_save_path: Path = MODEL_WEIGHTS_PATH,
    head_epochs: int = 8,
    finetune_epochs: int = 12,
    batch_size: int = 32,
    head_lr: float = 1e-3,
    finetune_lr: float = 1e-4,
    dropout_rate: float = 0.35,
    unfreeze_layers: int = 30,
    weights: Optional[str] = "imagenet",
    optimize_threshold: bool = True,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Execute the complete two-phase transfer-learning training workflow.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    set_reproducible_seeds(seed)

    train_csv = processed_dir / "train_split.csv"
    val_csv = processed_dir / "val_split.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError(
            f"Split CSV files not found in {processed_dir}. "
            "Run `.venv/bin/python scripts/prepare_dataset.py` first."
        )

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    print(f"[Train] Loaded {len(train_df):,} training images and {len(val_df):,} validation images.")

    # Save sample augmentation preview from the first valid training image
    if len(train_df) > 0 and Path(train_df.iloc[0]["image_path"]).exists():
        aug_fig = save_augmentation_sample_grid(train_df.iloc[0]["image_path"])
        print(f"[Augmentation] Saved sample augmentation visualization to: {aug_fig}")

    class_weights = compute_class_weights(train_df["binary_label"])
    print(f"[Class Weights] Computed training class weights: {class_weights}")

    train_ds = build_tf_dataset(
        image_paths=train_df["image_path"].tolist(),
        labels=train_df["binary_label"].astype(int).tolist(),
        batch_size=batch_size,
        is_training=True,
        seed=seed,
    )
    val_ds = build_tf_dataset(
        image_paths=val_df["image_path"].tolist(),
        labels=val_df["binary_label"].astype(int).tolist(),
        batch_size=batch_size,
        is_training=False,
        seed=seed,
    )

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Phase 1] Building {MODEL_ARCHITECTURE} model (weights={weights}) and training classification head...")
    model = build_efficientnetb0_binary_model(
        input_shape=INPUT_SHAPE,
        dropout_rate=dropout_rate,
        learning_rate=head_lr,
        weights=weights,
    )

    callbacks_phase1 = [
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=head_epochs,
        class_weight=class_weights,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    history_ft_dict = None
    unfrozen_count = 0
    if finetune_epochs > 0 and unfreeze_layers > 0:
        print(f"[Phase 2] Unfreezing top {unfreeze_layers} backbone layers for fine-tuning (lr={finetune_lr})...")
        unfrozen_count = unfreeze_top_backbone_layers(
            model,
            num_layers_to_unfreeze=unfreeze_layers,
            fine_tune_lr=finetune_lr,
        )
        callbacks_phase2 = [
            ModelCheckpoint(
                filepath=str(model_save_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
        ]
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=finetune_epochs,
            class_weight=class_weights,
            callbacks=callbacks_phase2,
            verbose=1,
        )
        history_ft_dict = history_ft.history

    # Ensure best checkpoint is loaded (or saved if validation metric was flat on tiny debug runs)
    if model_save_path.exists():
        model = tf.keras.models.load_model(str(model_save_path))
    else:
        model.save(str(model_save_path))

    # Validation threshold analysis (on validation set ONLY)
    val_probs = model.predict(val_ds, verbose=0).ravel()
    val_true = val_df["binary_label"].astype(int).to_numpy()
    threshold_analysis = select_validation_threshold(val_true, val_probs)
    active_threshold = (
        threshold_analysis["selected_threshold"]
        if optimize_threshold
        else DEFAULT_DECISION_THRESHOLD
    )

    merged_history = merge_histories(history_head.history, history_ft_dict)
    history_df = pd.DataFrame(merged_history)
    history_df.index.name = "epoch"
    history_csv_path = HISTORY_DIR / "training_history.csv"
    history_json_path = HISTORY_DIR / "training_history.json"
    history_df.to_csv(history_csv_path)
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_history, f, indent=2)

    model_config = {
        "architecture": MODEL_ARCHITECTURE,
        "input_shape": list(INPUT_SHAPE),
        "supported_classes": ["Benign", "Malignant"],
        "class_encoding": {"0": "Benign", "1": "Malignant"},
        "positive_class": "Malignant",
        "decision_threshold": active_threshold,
        "default_threshold": DEFAULT_DECISION_THRESHOLD,
        "threshold_analysis": threshold_analysis,
        "training_hyperparameters": {
            "random_seed": seed,
            "batch_size": batch_size,
            "head_epochs_requested": head_epochs,
            "finetune_epochs_requested": finetune_epochs,
            "total_epochs_completed": len(history_df),
            "head_learning_rate": head_lr,
            "finetune_learning_rate": finetune_lr,
            "dropout_rate": dropout_rate,
            "unfrozen_backbone_layers": unfrozen_count,
            "pretrained_weights": weights,
            "loss_function": "binary_crossentropy",
            "optimizer": "Adam",
            "class_weights": {str(k): v for k, v in class_weights.items()},
        },
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_weights_file": model_save_path.name,
    }

    with open(MODEL_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)

    print(f"[Complete] Best model saved to: {model_save_path}")
    print(f"[Complete] Model configuration saved to: {MODEL_CONFIG_PATH}")
    print(f"[Complete] Training history saved to: {history_csv_path}")
    return model_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 binary skin lesion classifier on HAM10000.")
    parser.add_argument("--processed-dir", type=Path, default=DATA_PROCESSED_DIR)
    parser.add_argument("--model-out", type=Path, default=MODEL_WEIGHTS_PATH)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--weights", type=str, default="imagenet", help="Use 'imagenet' or 'none'")
    parser.add_argument("--fixed-threshold", action="store_true", help="Keep decision threshold fixed at 0.50")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    weights_arg = None if args.weights.lower() == "none" else args.weights

    try:
        run_training(
            processed_dir=args.processed_dir,
            model_save_path=args.model_out,
            head_epochs=args.head_epochs,
            finetune_epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            head_lr=args.head_lr,
            finetune_lr=args.finetune_lr,
            dropout_rate=args.dropout,
            unfreeze_layers=args.unfreeze_layers,
            weights=weights_arg,
            optimize_threshold=not args.fixed_threshold,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"[Error] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
