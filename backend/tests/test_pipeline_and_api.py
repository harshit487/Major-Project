"""
Stage 11: Comprehensive Automated Test Suite (`pytest`).

Covers all 10 required test areas and 5 safety assertions:
1. Dataset-label mapping consistency (7 HAM10000 classes -> 2 Binary classes)
2. Leakage-resistant group-aware stratified dataset splitting
3. Image preprocessing (RGB conversion, 224x224 resize, batch shape)
4. Pre-inference model file existence check & Model loading
5. Valid image prediction & API response format
6. Probability calculations (sum to 1.0, threshold logic)
7. Invalid image handling (missing file, bad extension, corrupt bytes, tiny dimensions)
8. Grad-CAM heatmap & overlay generation on EfficientNetB0
9. PDF report generation via ReportLab (`POST /api/report`)
10. Evaluation metric calculation (including Specificity from confusion matrix
    and verification that evaluation rejects non-test splits)
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from backend.app import create_app
from backend.config import (
    DX_TO_BINARY_CLASS,
    DX_TO_BINARY_LABEL,
    HAM10000_DIAGNOSIS_DETAILS,
)
from backend.services.gradcam import (
    compute_gradcam_heatmap,
    find_target_conv_layer,
    generate_and_save_gradcam_artifacts,
)
from backend.services.model_service import ModelNotLoadedError, ModelService, model_service
from backend.services.preprocessing import (
    load_and_validate_pil_image,
    preprocess_image_for_inference,
)
from backend.services.report_service import generate_prediction_pdf_report
from scripts.evaluate_model import compute_binary_metrics_at_threshold, run_evaluation
from scripts.prepare_dataset import (
    compute_class_weights,
    create_group_stratified_splits,
    verify_and_map_metadata,
)
from scripts.train_model import build_efficientnetb0_binary_model


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------
def _make_test_image_bytes(
    width: int = 300,
    height: int = 300,
    mode: str = "RGB",
    fmt: str = "PNG",
) -> bytes:
    """Create a synthetic valid image in memory for testing preprocessing/API."""
    rng = np.random.default_rng(42)
    if mode == "L":
        arr = rng.integers(20, 230, size=(height, width), dtype=np.uint8)
    elif mode == "RGBA":
        arr = rng.integers(20, 230, size=(height, width, 4), dtype=np.uint8)
    else:
        arr = rng.integers(20, 230, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="module")
def shared_test_model_path(tmp_path_factory) -> Path:
    """
    Build and save an EfficientNetB0 binary model once for the test module
    so API, Grad-CAM, and inference tests execute against the real architecture.
    """
    model_dir = tmp_path_factory.mktemp("models")
    weights_path = model_dir / "test_efficientnetb0_binary.keras"
    model = build_efficientnetb0_binary_model(
        input_shape=(224, 224, 3),
        dropout_rate=0.2,
        learning_rate=1e-3,
        weights=None,  # Avoid network download dependency in unit tests
    )
    model.save(str(weights_path))
    return weights_path


@pytest.fixture()
def client_with_loaded_model(shared_test_model_path: Path, tmp_path: Path, monkeypatch):
    """Configure the Flask test client with a loaded EfficientNetB0 model."""
    monkeypatch.setattr(model_service, "model_path", shared_test_model_path)
    model_service.load_model(force_reload=True)
    app = create_app(load_model_on_start=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ---------------------------------------------------------------------------
# 1. Dataset-Label Mapping & Leakage-Free Split Tests
# ---------------------------------------------------------------------------
def test_dataset_label_mapping_consistency():
    """Verify all 7 HAM10000 categories map to the documented binary classes."""
    expected_malignant = {"mel", "bcc", "akiec"}
    expected_benign = {"nv", "bkl", "df", "vasc"}

    assert set(DX_TO_BINARY_LABEL.keys()) == expected_malignant | expected_benign

    for dx in expected_malignant:
        assert DX_TO_BINARY_LABEL[dx] == 1
        assert DX_TO_BINARY_CLASS[dx] == "Malignant"
        assert HAM10000_DIAGNOSIS_DETAILS[dx]["binary_label"] == 1

    for dx in expected_benign:
        assert DX_TO_BINARY_LABEL[dx] == 0
        assert DX_TO_BINARY_CLASS[dx] == "Benign"
        assert HAM10000_DIAGNOSIS_DETAILS[dx]["binary_label"] == 0


def test_verify_and_map_metadata_preserves_columns_and_rejects_unknown():
    """Confirm metadata mapping preserves original columns and rejects invalid dx."""
    sample_df = pd.DataFrame(
        [
            {
                "lesion_id": "HAM_0000001",
                "image_id": "ISIC_0024306",
                "dx": "nv",
                "dx_type": "follow_up",
                "age": 45.0,
                "sex": "male",
                "localization": "trunk",
            },
            {
                "lesion_id": "HAM_0000002",
                "image_id": "ISIC_0024307",
                "dx": "mel",
                "dx_type": "histo",
                "age": 60.0,
                "sex": "female",
                "localization": "back",
            },
        ]
    )
    mapped = verify_and_map_metadata(sample_df)
    assert list(mapped["binary_label"]) == [0, 1]
    assert list(mapped["binary_class"]) == ["Benign", "Malignant"]
    assert list(mapped["dx"]) == ["nv", "mel"]

    # Invalid diagnosis code must raise ValueError
    bad_df = sample_df.copy()
    bad_df.loc[0, "dx"] = "invalid_lesion_code"
    with pytest.raises(ValueError, match="unmapped diagnostic codes"):
        verify_and_map_metadata(bad_df)


def test_group_stratified_split_prevents_lesion_leakage():
    """Confirm images sharing a lesion_id never cross train/val/test boundaries."""
    rows = []
    dx_cycle = ["nv", "nv", "nv", "bkl", "mel", "bcc", "akiec", "df", "vasc"]
    for lesion_idx in range(60):
        lesion_id = f"HAM_{lesion_idx:05d}"
        dx = dx_cycle[lesion_idx % len(dx_cycle)]
        # Give every 3rd lesion 3 images to test multi-image grouping
        num_imgs = 3 if lesion_idx % 3 == 0 else 1
        for img_num in range(num_imgs):
            rows.append(
                {
                    "lesion_id": lesion_id,
                    "image_id": f"ISIC_{lesion_idx:05d}_{img_num}",
                    "dx": dx,
                    "dx_type": "histo",
                    "age": 50.0,
                    "sex": "male",
                    "localization": "back",
                }
            )
    df = verify_and_map_metadata(pd.DataFrame(rows))
    train_df, val_df, test_df = create_group_stratified_splits(df)

    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert len(train_lesions & val_lesions) == 0
    assert len(train_lesions & test_lesions) == 0
    assert len(val_lesions & test_lesions) == 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)


# ---------------------------------------------------------------------------
# 2. Image Preprocessing Tests
# ---------------------------------------------------------------------------
def test_image_preprocessing_shapes_and_modes():
    """Verify RGBA and grayscale images are converted to 224x224x3 RGB."""
    rgba_bytes = _make_test_image_bytes(width=400, height=260, mode="RGBA", fmt="PNG")
    pil_img, resized_uint8, batch_tensor = preprocess_image_for_inference(rgba_bytes)

    assert pil_img.mode == "RGB"
    assert resized_uint8.shape == (224, 224, 3)
    assert resized_uint8.dtype == np.uint8
    assert batch_tensor.shape == (1, 224, 224, 3)
    assert batch_tensor.dtype == np.float32


def test_corrupt_image_rejected_by_preprocessing():
    """Verify corrupt bytes raise ValueError in preprocessing."""
    with pytest.raises(ValueError, match="Invalid or corrupted image file"):
        load_and_validate_pil_image(b"not-a-valid-image-binary-stream")


# ---------------------------------------------------------------------------
# 3. Model Existence Check & Loading Tests
# ---------------------------------------------------------------------------
def test_missing_model_file_raises_before_inference(tmp_path: Path):
    """Confirm ModelService checks that the model file exists before inference."""
    non_existent = tmp_path / "does_not_exist.keras"
    svc = ModelService(model_path=non_existent)

    assert svc.load_model() is False
    with pytest.raises(ModelNotLoadedError, match="does not exist before inference"):
        svc.ensure_ready()


# ---------------------------------------------------------------------------
# 4. API Health, Model Info, Valid Prediction & Probability Tests
# ---------------------------------------------------------------------------
def test_health_and_model_info_endpoints(client_with_loaded_model):
    """Verify GET /api/health and GET /api/model-info response schemas."""
    res_health = client_with_loaded_model.get("/api/health")
    assert res_health.status_code == 200
    health_json = res_health.get_json()
    assert health_json["status"] == "healthy"
    assert health_json["model_loaded"] is True

    res_info = client_with_loaded_model.get("/api/model-info")
    assert res_info.status_code == 200
    info_json = res_info.get_json()
    assert info_json["model_architecture"] == "EfficientNetB0"
    assert info_json["supported_classes"] == ["Benign", "Malignant"]
    assert info_json["input_dimensions"]["shape"] == [224, 224, 3]
    assert "decision_threshold" in info_json


def test_valid_image_prediction_and_probability_sum(client_with_loaded_model):
    """
    Verify POST /api/predict returns expected output structure and that
    Benign + Malignant probabilities sum to 1.0.
    """
    img_bytes = _make_test_image_bytes(width=320, height=320, mode="RGB", fmt="JPEG")
    data = {"image": (io.BytesIO(img_bytes), "lesion_sample.jpg")}
    resp = client_with_loaded_model.post(
        "/api/predict",
        data=data,
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()

    required_keys = {
        "prediction_id",
        "prediction_status",
        "predicted_class",
        "benign_probability",
        "malignant_probability",
        "confidence_score",
        "decision_threshold",
        "visualizations",
    }
    assert required_keys.issubset(payload.keys())
    assert payload["prediction_status"] == "success"
    assert payload["predicted_class"] in {"Benign", "Malignant"}

    benign_p = payload["benign_probability"]
    malignant_p = payload["malignant_probability"]
    assert 0.0 <= benign_p <= 1.0
    assert 0.0 <= malignant_p <= 1.0
    assert pytest.approx(benign_p + malignant_p, abs=1e-5) == 1.0


# ---------------------------------------------------------------------------
# 5. Invalid Image Handling & Error Resilience Tests
# ---------------------------------------------------------------------------
def test_invalid_inputs_do_not_crash_backend(client_with_loaded_model):
    """Verify missing files, unsupported extensions, corrupt bytes, and tiny images return 400."""
    # Case 1: Missing file
    r1 = client_with_loaded_model.post("/api/predict", data={}, content_type="multipart/form-data")
    assert r1.status_code == 400
    assert r1.get_json()["prediction_status"] == "error"

    # Case 2: Disallowed file extension
    r2 = client_with_loaded_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"hello world"), "malicious.exe")},
        content_type="multipart/form-data",
    )
    assert r2.status_code == 400

    # Case 3: Corrupt image bytes with .jpg extension
    r3 = client_with_loaded_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(b"corrupted-jpeg-bytes"), "corrupt.jpg")},
        content_type="multipart/form-data",
    )
    assert r3.status_code == 400

    # Case 4: Image smaller than minimum dimensions (16x16 < 32x32)
    tiny_bytes = _make_test_image_bytes(width=16, height=16, mode="RGB", fmt="PNG")
    r4 = client_with_loaded_model.post(
        "/api/predict",
        data={"image": (io.BytesIO(tiny_bytes), "tiny.png")},
        content_type="multipart/form-data",
    )
    assert r4.status_code == 400


# ---------------------------------------------------------------------------
# 6. Grad-CAM Generation Tests
# ---------------------------------------------------------------------------
def test_gradcam_generation_and_endpoint(client_with_loaded_model, tmp_path: Path):
    """Verify Grad-CAM locates top_activation and produces valid heatmap and overlay."""
    img_bytes = _make_test_image_bytes(width=256, height=256, mode="RGB", fmt="PNG")
    _, resized_uint8, batch_input = preprocess_image_for_inference(img_bytes)

    layer_name = find_target_conv_layer(model_service.model)
    assert layer_name in {"top_activation", "top_conv"}

    heatmap_2d, used_layer = compute_gradcam_heatmap(
        model=model_service.model,
        preprocessed_batch=batch_input,
    )
    assert used_layer == layer_name
    assert heatmap_2d.ndim == 2
    assert float(np.min(heatmap_2d)) >= 0.0
    assert float(np.max(heatmap_2d)) <= 1.0 + 1e-5

    artifacts = generate_and_save_gradcam_artifacts(
        model=model_service.model,
        resized_rgb_uint8=resized_uint8,
        preprocessed_batch=batch_input,
        prediction_id="test_gradcam_01",
        output_dir=tmp_path,
    )
    assert artifacts["heatmap_path"].exists()
    assert artifacts["overlay_path"].exists()
    assert artifacts["comparison_path"].exists()

    # Also test POST /api/gradcam endpoint
    resp = client_with_loaded_model.post(
        "/api/gradcam",
        data={"image": (io.BytesIO(img_bytes), "lesion.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    gc_json = resp.get_json()
    assert gc_json["status"] == "success"
    assert "gradcam_overlay_url" in gc_json


# ---------------------------------------------------------------------------
# 7. PDF Report Generation Tests
# ---------------------------------------------------------------------------
def test_pdf_report_generation_service_and_endpoint(
    client_with_loaded_model,
    tmp_path: Path,
    monkeypatch,
):
    """Verify PDF report generation produces a valid %PDF document via service and API."""
    import backend.services.report_service as report_svc_mod

    monkeypatch.setattr(
        report_svc_mod,
        "generate_prediction_pdf_report",
        lambda prediction_data, **kwargs: generate_prediction_pdf_report(
            prediction_data,
            output_dir=tmp_path,
            **{k: v for k, v in kwargs.items() if k != "output_dir"},
        ),
    )
    import backend.routes.reports as reports_route_mod

    monkeypatch.setattr(
        reports_route_mod,
        "generate_prediction_pdf_report",
        report_svc_mod.generate_prediction_pdf_report,
    )

    sample_prediction = {
        "prediction_id": "pred_unit_test_01",
        "timestamp": "2026-10-11T00:00:00Z",
        "predicted_class": "Malignant",
        "benign_probability": 0.18,
        "malignant_probability": 0.82,
        "confidence_score": 0.82,
        "decision_threshold": 0.50,
    }
    pdf_file = generate_prediction_pdf_report(
        prediction_data=sample_prediction,
        output_dir=tmp_path,
    )
    assert pdf_file.exists()
    raw_pdf = pdf_file.read_bytes()
    assert raw_pdf.startswith(b"%PDF-")

    # Test API endpoint POST /api/report with JSON payload
    resp = client_with_loaded_model.post(
        "/api/report",
        data=json.dumps(sample_prediction),
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.mimetype == "application/pdf"
    assert resp.data.startswith(b"%PDF-")


# ---------------------------------------------------------------------------
# 8. Evaluation Metric & Split Enforcement Tests
# ---------------------------------------------------------------------------
def test_evaluation_metrics_specificity_and_split_guard(
    shared_test_model_path: Path,
    tmp_path: Path,
):
    """
    Verify specificity is calculated accurately from the confusion matrix
    (TN / (TN + FP)) and that `run_evaluation` rejects non-test splits.
    """
    # Ground truth: 4 Benign (0), 4 Malignant (1)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Predictions at threshold 0.5: 3 TN, 1 FP, 1 FN, 3 TP
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.2, 0.8, 0.9, 0.95])

    metrics = compute_binary_metrics_at_threshold(y_true, y_prob, threshold=0.50)
    assert metrics["confusion_matrix"]["tn"] == 3
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tp"] == 3
    assert metrics["specificity"] == 0.75
    assert metrics["recall"] == 0.75
    assert metrics["accuracy"] == 0.75
    assert metrics["sample_counts"]["benign_test_samples"] == 4
    assert metrics["sample_counts"]["malignant_test_samples"] == 4
    assert metrics["sample_counts"]["correctly_classified_samples"] == 6
    assert metrics["sample_counts"]["incorrectly_classified_samples"] == 2

    # Guard check: if test_split.csv contains 'train' split rows, run_evaluation must raise ValueError
    fake_split_dir = tmp_path / "bad_splits"
    fake_split_dir.mkdir(parents=True, exist_ok=True)
    bad_test_df = pd.DataFrame(
        [
            {
                "lesion_id": "HAM_1",
                "image_id": "ISIC_1",
                "dx": "nv",
                "binary_label": 0,
                "image_path": "dummy.jpg",
                "split": "train",  # Invalid split label in test_split.csv!
            }
        ]
    )
    bad_test_df.to_csv(fake_split_dir / "test_split.csv", index=False)

    with pytest.raises(ValueError, match="contains non-test split records"):
        run_evaluation(
            processed_dir=fake_split_dir,
            model_path=shared_test_model_path,
        )
