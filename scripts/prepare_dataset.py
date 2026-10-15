"""
Stage 2 & Stage 3: Dataset Audit, Explicit Binary-Label Mapping, and
Leakage-Resistant Group-Aware Stratified Splitting for HAM10000.

This module implements all 11 steps of the data preparation pipeline:
1. Load dataset metadata and verify expected columns.
2. Verify that corresponding dermoscopic image files exist on disk.
3. Identify missing or 'unknown' metadata values.
4. Check for duplicate image_id records and duplicate files.
5. Analyze the distribution of the original 7 diagnostic categories.
6. Map the 7 categories into Binary classes (0: Benign, 1: Malignant) with
   explicit clinical rationale and ambiguity notes.
7. Display the resulting binary-class distribution.
8. Quantify class imbalance and compute balanced class weights.
9. Inspect multi-image lesions (multiple image_ids per lesion_id).
10. Create 70% Train / 15% Validation / 15% Test splits using group-aware
    stratification on lesion_id so zero lesions cross split boundaries.
11. Save split CSVs, class_mapping.json, dataset_audit_summary.json, and
    class distribution figures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Ensure repository root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    CLASS_MAPPING_PATH,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DATASET_AUDIT_PATH,
    DX_TO_BINARY_CLASS,
    DX_TO_BINARY_LABEL,
    EXPECTED_METADATA_COLUMNS,
    FIGURES_DIR,
    HAM10000_DIAGNOSIS_DETAILS,
    RANDOM_SEED,
    TEST_SPLIT_RATIO,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
)


def save_class_mapping_artifact(output_path: Path = CLASS_MAPPING_PATH) -> Dict[str, Any]:
    """
    Export the explicit 7-class to binary-class mapping table and clinical
    documentation to JSON for auditing and reproducibility.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "HAM10000 (Human Against Machine with 10000 training images)",
        "reference": (
            "Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, "
            "a large collection of multi-source dermatoscopic images of common "
            "pigmented skin lesions. Sci. Data 5, 180161 (2018)."
        ),
        "target_task": "Binary Classification (Benign vs. Malignant)",
        "binary_encoding": {
            "0": "Benign (Non-cancerous)",
            "1": "Malignant (Cancerous / In-situ Neoplastic)",
        },
        "positive_class": {
            "label": 1,
            "name": "Malignant",
            "original_categories": ["mel", "bcc", "akiec"],
        },
        "negative_class": {
            "label": 0,
            "name": "Benign",
            "original_categories": ["nv", "bkl", "df", "vasc"],
        },
        "mapping_table": HAM10000_DIAGNOSIS_DETAILS,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def locate_metadata_csv(raw_dir: Path = DATA_RAW_DIR) -> Optional[Path]:
    """Search for HAM10000_metadata.csv inside data/raw/ or its subdirectories."""
    candidates = [
        raw_dir / "HAM10000_metadata.csv",
        raw_dir / "HAM10000_metadata",
        raw_dir / "HAM10000_metadata.tab",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    for found in raw_dir.rglob("*metadata*.csv"):
        if found.is_file():
            return found
    return None


def index_image_files(raw_dir: Path = DATA_RAW_DIR) -> Dict[str, Path]:
    """
    Recursively scan data/raw/ for .jpg/.jpeg/.png dermoscopic images and map
    image_id (filename stem, e.g. 'ISIC_0027419') -> Path.
    """
    image_index: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for img_path in raw_dir.rglob(ext):
            if img_path.is_file():
                image_index[img_path.stem] = img_path.resolve()
    return image_index


def verify_and_map_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate expected HAM10000 metadata columns and apply the explicit
    7-to-2 binary diagnostic label mapping while preserving original columns.
    """
    missing_cols = [c for c in EXPECTED_METADATA_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Metadata file is missing expected HAM10000 columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["dx"] = out["dx"].astype(str).str.strip().str.lower()

    unknown_dx = set(out["dx"].unique()) - set(DX_TO_BINARY_LABEL.keys())
    if unknown_dx:
        raise ValueError(
            f"Encountered unmapped diagnostic codes in 'dx' column: {sorted(unknown_dx)}. "
            f"Valid HAM10000 codes are: {sorted(DX_TO_BINARY_LABEL.keys())}"
        )

    out["binary_label"] = out["dx"].map(DX_TO_BINARY_LABEL).astype(int)
    out["binary_class"] = out["dx"].map(DX_TO_BINARY_CLASS)
    out["diagnosis_full_name"] = out["dx"].map(
        lambda x: HAM10000_DIAGNOSIS_DETAILS[x]["full_name"]
    )
    return out


def compute_class_weights(labels: np.ndarray | pd.Series) -> Dict[int, float]:
    """
    Compute balanced inverse-frequency class weights:
        w_c = N_total / (2 * N_c)
    to counter class imbalance during training without fabricating samples.
    """
    y = np.asarray(labels, dtype=int)
    total = len(y)
    n_benign = int(np.sum(y == 0))
    n_malignant = int(np.sum(y == 1))
    if n_benign == 0 or n_malignant == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: round(total / (2.0 * n_benign), 6),
        1: round(total / (2.0 * n_malignant), 6),
    }


def create_group_stratified_splits(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_SPLIT_RATIO,
    val_ratio: float = VAL_SPLIT_RATIO,
    test_ratio: float = TEST_SPLIT_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into 70% Train, 15% Validation, and 15% Test sets while:
    1. Preventing data leakage: all images with the same `lesion_id` are strictly
       assigned to the same split.
    2. Preserving class proportions across splits (stratified by original `dx`
       and `binary_label`).

    Implementation uses StratifiedGroupKFold (20 folds -> 5% per fold:
    14 folds = 70% Train, 3 folds = 15% Validation, 3 folds = 15% Test).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    n_lesions = df["lesion_id"].nunique()
    n_splits = 20 if n_lesions >= 20 else max(3, n_lesions)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    # Stratify on `dx` (which automatically stratifies `binary_label` as well)
    stratify_target = df["dx"] if df["dx"].value_counts().min() >= n_splits else df["binary_label"]
    groups = df["lesion_id"]

    fold_indices = list(sgkf.split(df, stratify_target, groups))

    if n_splits == 20:
        # 14 folds (70%) train, 3 folds (15%) val, 3 folds (15%) test
        train_folds = range(0, 14)
        val_folds = range(14, 17)
        test_folds = range(17, 20)
    else:
        n_test = max(1, int(round(n_splits * test_ratio)))
        n_val = max(1, int(round(n_splits * val_ratio)))
        n_train = max(1, n_splits - n_val - n_test)
        train_folds = range(0, n_train)
        val_folds = range(n_train, n_train + n_val)
        test_folds = range(n_train + n_val, n_splits)

    train_idx = np.concatenate([fold_indices[i][1] for i in train_folds])
    val_idx = np.concatenate([fold_indices[i][1] for i in val_folds])
    test_idx = np.concatenate([fold_indices[i][1] for i in test_folds])

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    # Strict verification of zero lesion leakage
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert len(train_lesions & val_lesions) == 0, "Data leakage detected between train and val!"
    assert len(train_lesions & test_lesions) == 0, "Data leakage detected between train and test!"
    assert len(val_lesions & test_lesions) == 0, "Data leakage detected between val and test!"

    return train_df, val_df, test_df


def plot_class_distributions(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_path: Path = FIGURES_DIR / "class_distribution.png",
) -> Path:
    """
    Generate and save a two-panel publication-quality figure showing:
    (a) Original 7-class distribution colored by Binary target mapping
    (b) Binary class distribution across Train, Validation, and Test splits
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)

    # Panel A: 7 original classes
    dx_counts = df["dx"].value_counts()
    dx_order = dx_counts.index.tolist()
    colors_7 = [
        "#dc2626" if DX_TO_BINARY_LABEL[dx] == 1 else "#0d9488" for dx in dx_order
    ]
    bars1 = axes[0].bar(dx_order, dx_counts.values, color=colors_7, edgecolor="#1e293b", alpha=0.9)
    axes[0].set_title("Original 7-Class Distribution (HAM10000)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Original Diagnostic Category (dx)", fontsize=10)
    axes[0].set_ylabel("Number of Dermoscopic Images", fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for bar in bars1:
        height = int(bar.get_height())
        axes[0].annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Legend for Panel A
    from matplotlib.patches import Patch
    axes[0].legend(
        handles=[
            Patch(facecolor="#0d9488", edgecolor="#1e293b", label="Mapped to Benign (0)"),
            Patch(facecolor="#dc2626", edgecolor="#1e293b", label="Mapped to Malignant (1)"),
        ],
        loc="upper right",
        frameon=True,
    )

    # Panel B: Binary distribution across splits
    splits = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    benign_counts = [
        int((train_df["binary_label"] == 0).sum()),
        int((val_df["binary_label"] == 0).sum()),
        int((test_df["binary_label"] == 0).sum()),
    ]
    malignant_counts = [
        int((train_df["binary_label"] == 1).sum()),
        int((val_df["binary_label"] == 1).sum()),
        int((test_df["binary_label"] == 1).sum()),
    ]

    x = np.arange(len(splits))
    width = 0.36
    b_bars = axes[1].bar(x - width / 2, benign_counts, width, label="Benign (0)", color="#0d9488", edgecolor="#1e293b")
    m_bars = axes[1].bar(x + width / 2, malignant_counts, width, label="Malignant (1)", color="#dc2626", edgecolor="#1e293b")

    axes[1].set_title("Binary Class Distribution Across Leakage-Free Splits", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(splits, fontsize=10)
    axes[1].set_ylabel("Number of Images", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(loc="upper right", frameon=True)

    for bar in list(b_bars) + list(m_bars):
        height = int(bar.get_height())
        axes[1].annotate(
            f"{height:,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_dataset_preparation(
    raw_dir: Path = DATA_RAW_DIR,
    processed_dir: Path = DATA_PROCESSED_DIR,
    require_images_on_disk: bool = True,
) -> Dict[str, Any]:
    """
    Execute the complete 11-step dataset audit, binary mapping, and
    group-stratified splitting pipeline.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    save_class_mapping_artifact(CLASS_MAPPING_PATH)

    metadata_path = locate_metadata_csv(raw_dir)
    if metadata_path is None:
        instructions = get_dataset_acquisition_instructions(raw_dir)
        raise FileNotFoundError(instructions)

    print(f"[Step 1] Loading metadata from: {metadata_path}")
    raw_df = pd.read_csv(metadata_path)
    total_metadata_rows = len(raw_df)
    print(f"         Loaded {total_metadata_rows:,} metadata records with columns: {list(raw_df.columns)}")

    # Step 2: Verify image files exist
    print(f"[Step 2] Indexing dermoscopic images in: {raw_dir}")
    image_index = index_image_files(raw_dir)
    raw_df["image_path"] = raw_df["image_id"].map(lambda i: str(image_index[i]) if i in image_index else "")
    raw_df["image_exists"] = raw_df["image_path"] != ""
    found_images_count = int(raw_df["image_exists"].sum())
    missing_images_count = int((~raw_df["image_exists"]).sum())
    print(f"         Verified {found_images_count:,} images present, {missing_images_count:,} missing.")

    if require_images_on_disk and found_images_count == 0:
        raise FileNotFoundError(
            f"Metadata file ({metadata_path.name}) was found with {total_metadata_rows:,} rows, "
            f"but 0 corresponding .jpg/.png image files were found in {raw_dir}.\n\n"
            + get_dataset_acquisition_instructions(raw_dir)
        )

    # Filter to existing images when require_images_on_disk is True
    working_df = raw_df[raw_df["image_exists"]].copy() if require_images_on_disk else raw_df.copy()

    # Step 3: Identify missing or 'unknown' metadata values
    print("[Step 3] Auditing missing and 'unknown' metadata fields...")
    null_counts = {col: int(working_df[col].isna().sum()) for col in EXPECTED_METADATA_COLUMNS}
    unknown_counts = {
        col: int((working_df[col].astype(str).str.lower() == "unknown").sum())
        for col in ["sex", "localization", "dx_type"]
    }

    # Step 4: Check for duplicate image_id records
    print("[Step 4] Checking for duplicate image_id entries...")
    duplicate_image_ids = int(working_df["image_id"].duplicated().sum())
    if duplicate_image_ids > 0:
        working_df = working_df.drop_duplicates(subset=["image_id"], keep="first").copy()

    # Step 5 & 6: Analyze 7 classes and map to Binary
    print("[Step 5 & 6] Mapping 7 HAM10000 categories to Binary (0: Benign, 1: Malignant)...")
    working_df = verify_and_map_metadata(working_df)

    seven_class_dist = {
        dx: {
            "count": int((working_df["dx"] == dx).sum()),
            "percentage": round(float((working_df["dx"] == dx).mean() * 100.0), 2),
            "mapped_binary_label": DX_TO_BINARY_LABEL[dx],
            "mapped_binary_class": DX_TO_BINARY_CLASS[dx],
            "full_name": HAM10000_DIAGNOSIS_DETAILS[dx]["full_name"],
        }
        for dx in sorted(DX_TO_BINARY_LABEL.keys())
    }

    # Step 7 & 8: Binary distribution and class imbalance analysis
    n_total = len(working_df)
    n_benign = int((working_df["binary_label"] == 0).sum())
    n_malignant = int((working_df["binary_label"] == 1).sum())
    imbalance_ratio = round(n_benign / max(1, n_malignant), 4)
    full_class_weights = compute_class_weights(working_df["binary_label"])

    print(
        f"[Step 7 & 8] Binary Distribution -> Benign (0): {n_benign:,} "
        f"({n_benign / n_total * 100:.2f}%), Malignant (1): {n_malignant:,} "
        f"({n_malignant / n_total * 100:.2f}%), Imbalance Ratio: {imbalance_ratio}:1"
    )

    # Step 9: Multi-image lesion inspection
    images_per_lesion = working_df.groupby("lesion_id")["image_id"].count()
    unique_lesions = int(images_per_lesion.shape[0])
    multi_image_lesions = int((images_per_lesion > 1).sum())
    max_images_for_single_lesion = int(images_per_lesion.max())
    conflicting_lesions = int(
        (working_df.groupby("lesion_id")["binary_label"].nunique() > 1).sum()
    )
    print(
        f"[Step 9] Multi-image Lesion Audit -> {unique_lesions:,} unique lesion_ids across "
        f"{n_total:,} images ({multi_image_lesions:,} lesions have >=2 images; "
        f"max {max_images_for_single_lesion} images/lesion)."
    )

    # Step 10: Leakage-resistant group-stratified splitting
    print("[Step 10] Creating group-aware stratified splits (70% Train / 15% Val / 15% Test)...")
    train_df, val_df, test_df = create_group_stratified_splits(working_df)
    train_class_weights = compute_class_weights(train_df["binary_label"])

    # Step 11: Save split CSVs, audit summary, and figures
    print("[Step 11] Saving split CSV files, audit JSON, and distribution plots...")
    train_csv_path = processed_dir / "train_split.csv"
    val_csv_path = processed_dir / "val_split.csv"
    test_csv_path = processed_dir / "test_split.csv"
    full_csv_path = processed_dir / "full_metadata_mapped.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(full_csv_path, index=False)

    fig_path = plot_class_distributions(working_df, train_df, val_df, test_df)

    def _split_summary(split_df: pd.DataFrame) -> Dict[str, Any]:
        total_s = len(split_df)
        benign_s = int((split_df["binary_label"] == 0).sum())
        malignant_s = int((split_df["binary_label"] == 1).sum())
        return {
            "total_images": total_s,
            "unique_lesions": int(split_df["lesion_id"].nunique()),
            "benign_count": benign_s,
            "malignant_count": malignant_s,
            "benign_pct": round(benign_s / max(1, total_s) * 100.0, 2),
            "malignant_pct": round(malignant_s / max(1, total_s) * 100.0, 2),
            "dx_breakdown": {
                dx: int((split_df["dx"] == dx).sum())
                for dx in sorted(DX_TO_BINARY_LABEL.keys())
            },
        }

    audit_summary: Dict[str, Any] = {
        "dataset_name": "HAM10000",
        "metadata_source": metadata_path.name,
        "total_metadata_rows": total_metadata_rows,
        "verified_images_count": found_images_count,
        "missing_images_count": missing_images_count,
        "duplicate_image_ids_removed": duplicate_image_ids,
        "missing_metadata_values": null_counts,
        "unknown_category_values": unknown_counts,
        "original_seven_class_distribution": seven_class_dist,
        "binary_class_distribution": {
            "Benign": {
                "label": 0,
                "count": n_benign,
                "percentage": round(n_benign / max(1, n_total) * 100.0, 2),
                "included_dx": ["nv", "bkl", "df", "vasc"],
            },
            "Malignant": {
                "label": 1,
                "count": n_malignant,
                "percentage": round(n_malignant / max(1, n_total) * 100.0, 2),
                "included_dx": ["mel", "bcc", "akiec"],
            },
        },
        "class_imbalance": {
            "benign_to_malignant_ratio": imbalance_ratio,
            "full_dataset_class_weights": full_class_weights,
            "training_set_class_weights": train_class_weights,
        },
        "lesion_group_analysis": {
            "unique_lesion_ids": unique_lesions,
            "lesions_with_single_image": int((images_per_lesion == 1).sum()),
            "lesions_with_multiple_images": multi_image_lesions,
            "max_images_per_lesion": max_images_for_single_lesion,
            "lesions_with_conflicting_binary_labels": conflicting_lesions,
            "zero_lesion_leakage_verified": True,
        },
        "splits": {
            "train": _split_summary(train_df),
            "val": _split_summary(val_df),
            "test": _split_summary(test_df),
        },
        "saved_artifacts": {
            "train_split_csv": str(train_csv_path.relative_to(PROJECT_ROOT)),
            "val_split_csv": str(val_csv_path.relative_to(PROJECT_ROOT)),
            "test_split_csv": str(test_csv_path.relative_to(PROJECT_ROOT)),
            "full_mapped_csv": str(full_csv_path.relative_to(PROJECT_ROOT)),
            "class_mapping_json": str(CLASS_MAPPING_PATH.relative_to(PROJECT_ROOT)),
            "class_distribution_figure": str(fig_path.relative_to(PROJECT_ROOT)),
        },
    }

    with open(DATASET_AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print(f"[Complete] Audit summary saved to: {DATASET_AUDIT_PATH}")
    return audit_summary


def get_dataset_acquisition_instructions(raw_dir: Path = DATA_RAW_DIR) -> str:
    """Return clear instructions for obtaining the official HAM10000 dataset."""
    return (
        "========================================================================\n"
        "OFFICIAL HAM10000 DATASET NOT FOUND IN data/raw/\n"
        "========================================================================\n"
        f"Expected directory: {raw_dir}\n\n"
        "To run dataset preparation and model training on real HAM10000 data:\n"
        "1. Download the official HAM10000 dataset from one of the official sources:\n"
        "   - Harvard Dataverse (Official Primary Source):\n"
        "     https://doi.org/10.7910/DVN/DBW86T\n"
        "   - ISIC Archive (Challenge 2018 Task 3):\n"
        "     https://challenge.isic-archive.com/data/#2018\n"
        "   - Kaggle Mirror ('Skin Cancer MNIST: HAM10000'):\n"
        "     https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000\n\n"
        "2. Place the extracted files into data/raw/ so it contains:\n"
        "   data/raw/\n"
        "   ├── HAM10000_metadata.csv\n"
        "   ├── HAM10000_images_part_1/   (ISIC_0024306.jpg ...)\n"
        "   └── HAM10000_images_part_2/   (ISIC_0029306.jpg ...)\n"
        "   (Or place all .jpg images directly in data/raw/images/)\n\n"
        "3. Re-run:\n"
        "   .venv/bin/python scripts/prepare_dataset.py\n"
        "========================================================================"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit HAM10000 metadata, map 7 classes to binary, and create group-stratified splits."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DATA_RAW_DIR,
        help="Path to raw HAM10000 dataset directory (default: data/raw).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DATA_PROCESSED_DIR,
        help="Path to save processed splits and audit reports (default: data/processed).",
    )
    parser.add_argument(
        "--export-mapping-only",
        action="store_true",
        help="Export class_mapping.json and print mapping table without requiring dataset files.",
    )
    args = parser.parse_args()

    mapping_payload = save_class_mapping_artifact(CLASS_MAPPING_PATH)
    print(f"[Mapping] Exported explicit 7-to-2 class mapping table to: {CLASS_MAPPING_PATH}")

    if args.export_mapping_only:
        return

    try:
        run_dataset_preparation(
            raw_dir=args.raw_dir,
            processed_dir=args.processed_dir,
            require_images_on_disk=True,
        )
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
