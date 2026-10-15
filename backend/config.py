"""
Centralized configuration for Skin Disease Classification Using Deep Learning
on the HAM10000 Dataset.

All paths are resolved dynamically via pathlib.Path relative to the repository
root—no local absolute paths are hardcoded.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

# ---------------------------------------------------------------------------
# Directory Hierarchy (Relative to Repository Root)
# ---------------------------------------------------------------------------
BACKEND_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = BACKEND_DIR.parent

DATA_DIR: Path = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", BACKEND_DIR / "models"))
RESULTS_DIR: Path = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results"))
FIGURES_DIR: Path = RESULTS_DIR / "figures"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
HISTORY_DIR: Path = RESULTS_DIR / "training_history"

REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", PROJECT_ROOT / "reports"))
STATIC_DIR: Path = BACKEND_DIR / "static"
UPLOADS_DIR: Path = STATIC_DIR / "uploads"
GRADCAM_DIR: Path = STATIC_DIR / "gradcam"

# Ensure runtime directories exist
for _directory in [
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    HISTORY_DIR,
    REPORTS_DIR,
    UPLOADS_DIR,
    GRADCAM_DIR,
]:
    _directory.mkdir(parents=True, exist_ok=True)

# Ensure Keras & Matplotlib cache directories are writable inside the project
_KERAS_CACHE_DIR: Path = MODELS_DIR / ".keras_cache"
_MPL_CACHE_DIR: Path = RESULTS_DIR / ".mpl_cache"
_KERAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(_KERAS_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR))

# ---------------------------------------------------------------------------
# Model & Training Hyperparameters
# ---------------------------------------------------------------------------
MODEL_ARCHITECTURE: str = "EfficientNetB0"
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
IMAGE_SIZE: tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)
INPUT_SHAPE: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
DEFAULT_DECISION_THRESHOLD: float = float(os.getenv("DECISION_THRESHOLD", "0.50"))

# Split ratios (Group-aware stratified split by lesion_id)
TRAIN_SPLIT_RATIO: float = 0.70
VAL_SPLIT_RATIO: float = 0.15
TEST_SPLIT_RATIO: float = 0.15

# Default artifact file paths
MODEL_WEIGHTS_PATH: Path = Path(
    os.getenv("MODEL_WEIGHTS_PATH", MODELS_DIR / "efficientnetb0_ham10000_binary.keras")
)
MODEL_CONFIG_PATH: Path = MODELS_DIR / "model_config.json"
EVALUATION_METRICS_PATH: Path = METRICS_DIR / "evaluation_metrics.json"
DATASET_AUDIT_PATH: Path = DATA_PROCESSED_DIR / "dataset_audit_summary.json"
CLASS_MAPPING_PATH: Path = DATA_PROCESSED_DIR / "class_mapping.json"

# ---------------------------------------------------------------------------
# HAM10000 Explicit Diagnostic Mapping Table (7 Classes -> Binary)
# ---------------------------------------------------------------------------
# Reference: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset,
# a large collection of multi-source dermatoscopic images of common pigmented
# skin lesions. Sci. Data 5, 180161 (2018).
HAM10000_DIAGNOSIS_DETAILS: Dict[str, Dict[str, Any]] = {
    "akiec": {
        "original_code": "akiec",
        "full_name": "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
        "binary_label": 1,
        "binary_class": "Malignant",
        "clinical_category": "Malignant / Pre-malignant Squamous Neoplasm (In Situ)",
        "clinical_rationale": (
            "Includes Bowen's disease (squamous cell carcinoma in situ) and actinic "
            "keratoses (intraepidermal squamous dysplasia capable of progressing to "
            "invasive squamous cell carcinoma). In clinical dermoscopic triage and "
            "standard HAM10000 binary benchmarks, akiec is grouped with malignant "
            "lesions because it requires clinical intervention, biopsy, or ablation."
        ),
        "ambiguity_note": (
            "Ambiguous boundary: Early actinic keratosis is clinically considered "
            "precancerous dysplasia rather than invasive cancer, whereas Bowen's "
            "disease in the same HAM10000 category is carcinoma in situ. Grouping "
            "akiec as Malignant prioritizes patient safety (reducing false negatives "
            "for squamous neoplasia)."
        ),
    },
    "bcc": {
        "original_code": "bcc",
        "full_name": "Basal cell carcinoma",
        "binary_label": 1,
        "binary_class": "Malignant",
        "clinical_category": "Malignant Epithelial Neoplasm",
        "clinical_rationale": (
            "Basal cell carcinoma is the most common malignant skin cancer worldwide. "
            "Though rarely metastatic, it is locally invasive and destructive, "
            "requiring surgical excision or targeted therapy."
        ),
        "ambiguity_note": "Unambiguous malignant cutaneous neoplasm.",
    },
    "mel": {
        "original_code": "mel",
        "full_name": "Melanoma",
        "binary_label": 1,
        "binary_class": "Malignant",
        "clinical_category": "Malignant Melanocytic Neoplasm",
        "clinical_rationale": (
            "Malignant melanoma is the most lethal form of skin cancer with high "
            "metastatic potential if not detected early."
        ),
        "ambiguity_note": "Unambiguous malignant melanocytic neoplasm.",
    },
    "bkl": {
        "original_code": "bkl",
        "full_name": "Benign keratosis-like lesions (solar lentigines / seborrheic keratoses / lichen-planus like keratoses)",
        "binary_label": 0,
        "binary_class": "Benign",
        "clinical_category": "Benign Epidermal Proliferation",
        "clinical_rationale": (
            "Encompasses seborrheic keratoses, solar lentigines, and lichen-planus-like "
            "keratoses, all of which are biologically benign epidermal lesions with "
            "no malignant potential."
        ),
        "ambiguity_note": (
            "While visually capable of mimicking melanoma under dermoscopy (especially "
            "inflamed seborrheic keratosis or LPLK), histopathologically and clinically "
            "these lesions are strictly benign."
        ),
    },
    "df": {
        "original_code": "df",
        "full_name": "Dermatofibroma",
        "binary_label": 0,
        "binary_class": "Benign",
        "clinical_category": "Benign Fibrohistiocytic Proliferation",
        "clinical_rationale": (
            "Dermatofibroma is a common, benign dermal nodule composed of fibroblastic "
            "and histiocytic cells requiring no oncological excision."
        ),
        "ambiguity_note": "Unambiguous benign cutaneous nodule.",
    },
    "nv": {
        "original_code": "nv",
        "full_name": "Melanocytic nevi",
        "binary_label": 0,
        "binary_class": "Benign",
        "clinical_category": "Benign Melanocytic Neoplasm",
        "clinical_rationale": (
            "Common benign proliferations of melanocytes (moles). Represents the "
            "majority (~67%) of the HAM10000 dataset."
        ),
        "ambiguity_note": (
            "Dysplastic/atypical nevi may clinically resemble early melanoma, but all "
            "lesions labeled 'nv' in HAM10000 were verified as benign via follow-up, "
            "expert consensus, confocal microscopy, or histopathology."
        ),
    },
    "vasc": {
        "original_code": "vasc",
        "full_name": "Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas, hemorrhage)",
        "binary_label": 0,
        "binary_class": "Benign",
        "clinical_category": "Benign Vascular Proliferation / Hemorrhage",
        "clinical_rationale": (
            "Includes cherry angiomas, angiokeratomas, pyogenic granulomas, and "
            "subcorneal hemorrhage—all non-malignant vascular conditions."
        ),
        "ambiguity_note": "Unambiguous benign vascular category in HAM10000.",
    },
}

# Direct lookup dictionaries
DX_TO_BINARY_LABEL: Dict[str, int] = {
    dx: info["binary_label"] for dx, info in HAM10000_DIAGNOSIS_DETAILS.items()
}

DX_TO_BINARY_CLASS: Dict[str, str] = {
    dx: info["binary_class"] for dx, info in HAM10000_DIAGNOSIS_DETAILS.items()
}

BINARY_CLASSES: Dict[int, str] = {
    0: "Benign",
    1: "Malignant",
}

EXPECTED_METADATA_COLUMNS: list[str] = [
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
]

# ---------------------------------------------------------------------------
# API & File Upload Security Settings
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MIN_IMAGE_DIMENSION: int = 32
MAX_IMAGE_DIMENSION: int = 8192

CORS_ORIGINS: list[str] = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000",
    ).split(",")
    if origin.strip()
]

# ---------------------------------------------------------------------------
# Academic Project & Medical Disclaimer Information
# ---------------------------------------------------------------------------
PROJECT_METADATA: Dict[str, Any] = {
    "title": "Skin Disease Classification Using Deep Learning on the HAM10000 Dataset",
    "institution": "Graphic Era Hill University, Dehradun, Uttarakhand",
    "department": "Department of Computer Science and Engineering",
    "faculty_guide": "Nishant Bhandari, Assistant Professor",
    "team_id": "CSE27-304",
    "team_members": [
        {"name": "Kamal Kant Bisht", "roll_no": "2318930"},
        {"name": "Lakshya Tiwari", "roll_no": "2319035"},
        {"name": "Harshit Srikoti", "roll_no": "2318847"},
        {"name": "Manish Tinsola", "roll_no": "2319073"},
    ],
}

MEDICAL_DISCLAIMER: str = (
    "MEDICAL DISCLAIMER: This software is an educational and academic research "
    "prototype developed at Graphic Era Hill University (Team CSE27-304). "
    "Predictions are experimental, uncalibrated estimates and may be incorrect. "
    "This system is NOT a clinically validated diagnostic device and is NOT a "
    "substitute for professional examination by a qualified dermatologist. "
    "A 'Benign' prediction does not rule out skin cancer, and a 'Malignant' "
    "prediction does not confirm skin cancer. Never make clinical or treatment "
    "decisions based on this application. Always consult a board-certified "
    "healthcare professional for any skin lesion concerns."
)


def get_active_decision_threshold() -> float:
    """
    Return the decision threshold saved in model_config.json (selected on the
    validation set), falling back to DEFAULT_DECISION_THRESHOLD (0.50).
    """
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                if "decision_threshold" in cfg:
                    return float(cfg["decision_threshold"])
        except Exception:
            pass
    return DEFAULT_DECISION_THRESHOLD
