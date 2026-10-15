"""
Stage 3: Reusable Image Preprocessing & Data Augmentation Pipeline.

Guarantees identical preprocessing during training, evaluation, and real-time
API inference:
- Loads valid dermoscopic images via Pillow / TensorFlow.
- Converts to 3-channel RGB.
- Resizes to 224 x 224 pixels.
- Applies `tf.keras.applications.efficientnet.preprocess_input` (float32 [0, 255]
  as expected by Keras EfficientNetB0's built-in rescaling/normalization stem).
- Provides a training-only augmentation pipeline preserving lesion morphology
  and color semantics.
- Generates sample visualizations comparing original and augmented images.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import (
    FIGURES_DIR,
    IMAGE_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_SIZE,
    IMAGE_WIDTH,
    RANDOM_SEED,
)


def load_and_validate_pil_image(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image],
) -> Image.Image:
    """
    Safely load an image from a path, raw bytes, stream, or PIL.Image instance,
    verify its integrity, and convert it to a 3-channel RGB PIL Image.
    """
    if isinstance(source, Image.Image):
        img = source.copy()
        img.load()
        return img.convert("RGB")

    try:
        if isinstance(source, (bytes, bytearray)):
            stream = io.BytesIO(source)
            with Image.open(stream) as opened:
                opened.verify()
            stream.seek(0)
            with Image.open(stream) as reopened:
                reopened.load()
                return reopened.convert("RGB")

        if isinstance(source, io.BytesIO):
            source.seek(0)
            with Image.open(source) as opened:
                opened.verify()
            source.seek(0)
            with Image.open(source) as reopened:
                reopened.load()
                return reopened.convert("RGB")

        path = Path(source)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")

        with Image.open(path) as opened:
            opened.verify()
        with Image.open(path) as reopened:
            reopened.load()
            return reopened.convert("RGB")

    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc


def preprocess_pil_image(
    pil_image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a PIL Image to RGB, resize to `target_size` (224 x 224), and apply
    EfficientNet preprocessing.

    Returns:
        resized_rgb_uint8: (H, W, 3) uint8 array in [0, 255] for visualization.
        model_input_batch: (1, H, W, 3) float32 array preprocessed for EfficientNetB0.
    """
    from tensorflow.keras.applications.efficientnet import preprocess_input

    rgb_img = pil_image.convert("RGB")
    resized_img = rgb_img.resize((target_size[1], target_size[0]), Image.Resampling.BILINEAR)
    resized_rgb_uint8 = np.asarray(resized_img, dtype=np.uint8)

    float_array = resized_rgb_uint8.astype(np.float32)
    preprocessed = preprocess_input(float_array)
    model_input_batch = np.expand_dims(preprocessed, axis=0)
    return resized_rgb_uint8, model_input_batch


def preprocess_image_for_inference(
    source: Union[str, Path, bytes, io.BytesIO, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """
    End-to-end preprocessing entry point for inference and testing.

    Returns:
        original_pil: Original RGB PIL Image.
        resized_rgb_uint8: (224, 224, 3) uint8 numpy array.
        model_input_batch: (1, 224, 224, 3) float32 tensor/array ready for model.predict().
    """
    original_pil = load_and_validate_pil_image(source)
    resized_rgb_uint8, model_input_batch = preprocess_pil_image(original_pil, target_size)
    return original_pil, resized_rgb_uint8, model_input_batch


def build_augmentation_layer(seed: int = RANDOM_SEED):
    """
    Build a Keras Sequential augmentation pipeline used ONLY on training samples.

    Augmentations are carefully chosen for planar dermoscopy:
    - Horizontal & vertical flips (dermoscopic views are rotationally invariant)
    - Small rotations (+/- 8% of 2*pi ~= +/- 28.8 degrees)
    - Moderate zoom (+/- 10%)
    - Small translations (+/- 5%)
    - Mild contrast (+/- 10%) and brightness (+/- 8%) adjustments clipped to [0, 255]
    """
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=seed),
            layers.RandomRotation(factor=0.08, fill_mode="reflect", seed=seed),
            layers.RandomZoom(
                height_factor=(-0.10, 0.10),
                width_factor=(-0.10, 0.10),
                fill_mode="reflect",
                seed=seed,
            ),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
                seed=seed,
            ),
            layers.RandomContrast(factor=0.10, seed=seed),
            layers.RandomBrightness(factor=0.08, value_range=(0.0, 255.0), seed=seed),
        ],
        name="dermoscopic_augmentation",
    )


def build_tf_dataset(
    image_paths: list[str],
    labels: list[int],
    batch_size: int = 32,
    is_training: bool = False,
    seed: int = RANDOM_SEED,
):
    """
    Construct a high-performance `tf.data.Dataset` pipeline for training,
    validation, or test evaluation.

    - Applies identical RGB decoding, 224x224 resizing, and `preprocess_input`.
    - Applies data augmentation ONLY when `is_training=True`.
    - Never augments validation or test splits.
    """
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    path_ds = tf.data.Dataset.from_tensor_slices(
        (
            tf.constant(image_paths, dtype=tf.string),
            tf.constant(labels, dtype=tf.float32),
        )
    )

    if is_training:
        path_ds = path_ds.shuffle(buffer_size=min(len(image_paths), 4096), seed=seed, reshuffle_each_iteration=True)

    aug_layer = build_augmentation_layer(seed=seed) if is_training else None

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        raw_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(raw_bytes, channels=IMAGE_CHANNELS, expand_animations=False)
        img.set_shape([None, None, IMAGE_CHANNELS])
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, tf.expand_dims(label, axis=-1)

    ds = path_ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)

    if is_training and aug_layer is not None:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(aug_layer(x, training=True), 0.0, 255.0), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return ds.prefetch(tf.data.AUTOTUNE)


def save_augmentation_sample_grid(
    image_path: Union[str, Path],
    output_path: Path = FIGURES_DIR / "augmentation_samples.png",
    num_variants: int = 5,
    seed: int = RANDOM_SEED,
) -> Path:
    """
    Generate and save a visual comparison showing the original resized
    dermoscopic image alongside `num_variants` random training augmentations.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _, resized_uint8, batch_input = preprocess_image_for_inference(image_path)
    aug_layer = build_augmentation_layer(seed=seed)

    cols = num_variants + 1
    fig, axes = plt.subplots(1, cols, figsize=(3.0 * cols, 3.4), dpi=150)

    axes[0].imshow(resized_uint8)
    axes[0].set_title("Original (224x224)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    tensor_in = tf.convert_to_tensor(batch_input, dtype=tf.float32)
    for idx in range(1, cols):
        aug_out = aug_layer(tensor_in, training=True)[0].numpy()
        aug_uint8 = np.clip(aug_out, 0, 255).astype(np.uint8)
        axes[idx].imshow(aug_uint8)
        axes[idx].set_title(f"Augmented #{idx}", fontsize=10)
        axes[idx].axis("off")

    plt.suptitle(
        "Dermoscopic Image Preprocessing & Training-Only Augmentation Preview",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
