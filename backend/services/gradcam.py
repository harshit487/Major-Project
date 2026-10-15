"""
Stage 6: Explainable AI (XAI) Service using Gradient-weighted Class Activation
Mapping (Grad-CAM) for EfficientNetB0.

Methodology:
1. Locates the final spatial convolutional feature map layer in EfficientNetB0
   (`top_activation` or `top_conv`, shape 7x7x1280 for 224x224 inputs).
2. Computes the gradient of the predicted class score with respect to the
   convolutional feature map activations using `tf.GradientTape`.
3. Global-average-pools the gradients across spatial dimensions (H, W) to obtain
   channel importance weights alpha_k.
4. Computes ReLU(sum_k alpha_k * A^k), normalizes to [0, 1], resizes to 224x224,
   and applies the 'jet' colormap using Matplotlib/Pillow.
5. Saves standalone heatmap, blended overlay, and side-by-side comparison images.

Important Clinical Caveat:
Grad-CAM highlights image regions associated with the deep learning model's
prediction; it does NOT establish a medical diagnosis or prove that the
highlighted region is clinically or histopathologically abnormal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image

from backend.config import GRADCAM_DIR, IMAGE_SIZE

GRADCAM_EXPLANATION_NOTE: str = (
    "Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes the spatial "
    "regions in the final convolutional layer (top_activation) of EfficientNetB0 "
    "that most strongly influenced the network's output. Warm colors (red/yellow) "
    "indicate higher relative activation importance for the model's prediction. "
    "This visualization is an interpretability aid for model auditing—it does NOT "
    "establish a clinical diagnosis or confirm anatomical pathology."
)


def find_target_conv_layer(model) -> str:
    """
    Locate the last 4D convolutional activation layer in EfficientNetB0.
    Prefers `top_activation` or `top_conv`, falling back to the last layer
    with a 4D spatial output tensor (batch, height, width, channels).
    """
    preferred_names = ["top_activation", "top_conv"]
    layer_names = {layer.name for layer in model.layers}
    for name in preferred_names:
        if name in layer_names:
            return name

    for layer in reversed(model.layers):
        try:
            output_shape = layer.output.shape
            if len(output_shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("Could not locate a 4D convolutional layer in the model for Grad-CAM.")


def compute_gradcam_heatmap(
    model,
    preprocessed_batch: np.ndarray,
    target_layer_name: Optional[str] = None,
    threshold: float = 0.50,
) -> Tuple[np.ndarray, str]:
    """
    Compute the normalized 2D Grad-CAM heatmap in [0, 1] for a (1, 224, 224, 3)
    preprocessed input tensor.

    Returns:
        heatmap_2d: (H_conv, W_conv) float32 array in [0, 1].
        used_layer_name: Name of the convolutional layer inspected.
    """
    import tensorflow as tf

    if target_layer_name is None:
        target_layer_name = find_target_conv_layer(model)

    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output],
    )

    input_tensor = tf.cast(preprocessed_batch, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_tensor, training=False)
        tape.watch(conv_outputs)
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]
        malignant_prob = predictions[:, 0]
        # Differentiate with respect to the predicted class score
        pred_is_malignant = float(malignant_prob.numpy()[0]) >= threshold
        target_score = malignant_prob if pred_is_malignant else (1.0 - malignant_prob)

    grads = tape.gradient(target_score, conv_outputs)
    if grads is None:
        raise RuntimeError(f"Grad-CAM gradient computation returned None for layer '{target_layer_name}'.")

    conv_map = conv_outputs[0]  # (H, W, C)
    grad_map = grads[0]         # (H, W, C)

    # Channel importance weights via spatial global average pooling
    pooled_grads = tf.reduce_mean(grad_map, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(conv_map, pooled_grads), axis=-1)

    # Apply ReLU; if all activations are negative, use absolute magnitude fallback
    cam_relu = tf.maximum(cam, 0.0).numpy()
    if float(np.max(cam_relu)) <= 1e-8:
        cam_relu = np.abs(cam.numpy())

    max_val = float(np.max(cam_relu))
    if max_val > 1e-8:
        heatmap = (cam_relu - float(np.min(cam_relu))) / (max_val - float(np.min(cam_relu)) + 1e-8)
    else:
        heatmap = np.zeros_like(cam_relu, dtype=np.float32)

    return heatmap.astype(np.float32), target_layer_name


def render_heatmap_and_overlay(
    original_rgb_uint8: np.ndarray,
    heatmap_2d: np.ndarray,
    alpha: float = 0.42,
) -> Tuple[Image.Image, Image.Image]:
    """
    Resize the 2D Grad-CAM heatmap to match `original_rgb_uint8` (224x224x3),
    apply the 'jet' colormap, and create both the standalone heatmap PIL Image
    and the alpha-blended overlay PIL Image.
    """
    import matplotlib

    h, w = original_rgb_uint8.shape[:2]
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_2d, 0.0, 1.0))
    heatmap_pil_gray = Image.fromarray(heatmap_uint8).resize((w, h), Image.Resampling.BICUBIC)
    heatmap_resized = np.asarray(heatmap_pil_gray, dtype=np.float32) / 255.0

    jet_cmap = matplotlib.colormaps["jet"]
    colored_rgba = jet_cmap(heatmap_resized)  # (H, W, 4) in [0, 1]
    colored_rgb_uint8 = np.uint8(255 * colored_rgba[:, :, :3])

    blended_float = (1.0 - alpha) * original_rgb_uint8.astype(np.float32) + alpha * colored_rgb_uint8.astype(np.float32)
    blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)

    return Image.fromarray(colored_rgb_uint8), Image.fromarray(blended_uint8)


def generate_and_save_gradcam_artifacts(
    model,
    resized_rgb_uint8: np.ndarray,
    preprocessed_batch: np.ndarray,
    prediction_id: str,
    output_dir: Path = GRADCAM_DIR,
    threshold: float = 0.50,
    alpha: float = 0.42,
) -> Dict[str, Any]:
    """
    Generate and save:
    - Original resized image (`{prediction_id}_original.png`)
    - Grad-CAM heatmap (`{prediction_id}_heatmap.png`)
    - Blended Grad-CAM overlay (`{prediction_id}_overlay.png`)
    - Side-by-side 3-panel comparison (`{prediction_id}_comparison.png`)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_2d, target_layer_name = compute_gradcam_heatmap(
        model=model,
        preprocessed_batch=preprocessed_batch,
        threshold=threshold,
    )
    heatmap_pil, overlay_pil = render_heatmap_and_overlay(
        original_rgb_uint8=resized_rgb_uint8,
        heatmap_2d=heatmap_2d,
        alpha=alpha,
    )
    original_pil = Image.fromarray(resized_rgb_uint8)

    original_path = output_dir / f"{prediction_id}_original.png"
    heatmap_path = output_dir / f"{prediction_id}_heatmap.png"
    overlay_path = output_dir / f"{prediction_id}_overlay.png"
    comparison_path = output_dir / f"{prediction_id}_comparison.png"

    original_pil.save(original_path, format="PNG")
    heatmap_pil.save(heatmap_path, format="PNG")
    overlay_pil.save(overlay_path, format="PNG")

    # Render side-by-side comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150)
    axes[0].imshow(resized_rgb_uint8)
    axes[0].set_title("Original Image (224x224)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(np.asarray(heatmap_pil))
    axes[1].set_title(f"Grad-CAM Heatmap ({target_layer_name})", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(np.asarray(overlay_pil))
    axes[2].set_title("Blended Grad-CAM Overlay", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    fig.savefig(comparison_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "prediction_id": prediction_id,
        "target_layer": target_layer_name,
        "original_path": original_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "comparison_path": comparison_path,
        "original_url": f"/api/static/gradcam/{original_path.name}",
        "heatmap_url": f"/api/static/gradcam/{heatmap_path.name}",
        "overlay_url": f"/api/static/gradcam/{overlay_path.name}",
        "comparison_url": f"/api/static/gradcam/{comparison_path.name}",
        "explanation": GRADCAM_EXPLANATION_NOTE,
    }
