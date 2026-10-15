import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  CheckCircle2,
  FileImage,
  Loader2,
  RefreshCw,
  Sparkles,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';
import { extractApiErrorMessage, predictLesionImage } from '../services/api';

const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png'];
const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

export default function ImageUploadPage({
  initialFile,
  onClearInitialFile,
  onPredictionSuccess,
  health,
}) {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState(null);
  const [validationError, setValidationError] = useState(null);
  const [apiError, setApiError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const validateAndSelectFile = (file) => {
    setValidationError(null);
    setApiError(null);

    if (!file) return;

    const ext = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : '';
    if (!ALLOWED_EXTENSIONS.includes(ext) && !ALLOWED_MIME_TYPES.includes(file.type)) {
      setValidationError(
        `Unsupported file format "${ext || file.type || 'unknown'}". Please select a JPG, JPEG, or PNG dermoscopic image.`
      );
      return;
    }

    if (file.size === 0) {
      setValidationError('The selected file is empty (0 bytes).');
      return;
    }

    if (file.size > MAX_SIZE_BYTES) {
      setValidationError(
        `File size (${(file.size / (1024 * 1024)).toFixed(2)} MB) exceeds the 10 MB upload limit.`
      );
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      if (img.width < 32 || img.height < 32) {
        URL.revokeObjectURL(objectUrl);
        setValidationError(
          `Image dimensions (${img.width}×${img.height}px) are too small. Minimum supported resolution is 32×32 pixels.`
        );
        return;
      }
      setImageDimensions({ width: img.width, height: img.height });
      setSelectedFile(file);
      setPreviewUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return objectUrl;
      });
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setValidationError(
        'Selected file could not be decoded as a valid image. Please upload an uncorrupted JPG or PNG file.'
      );
    };
    img.src = objectUrl;
  };

  useEffect(() => {
    if (initialFile) {
      validateAndSelectFile(initialFile);
      if (onClearInitialFile) onClearInitialFile();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialFile]);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleClearFile = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setImageDimensions(null);
    setValidationError(null);
    setApiError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleRunInference = async () => {
    if (!selectedFile) {
      setValidationError('Please select a dermoscopic image before running analysis.');
      return;
    }

    setIsSubmitting(true);
    setApiError(null);

    try {
      const result = await predictLesionImage(selectedFile, true);
      onPredictionSuccess({
        ...result,
        clientPreviewUrl: previewUrl,
        uploadedFileName: selectedFile.name,
        uploadedFileDimensions: imageDimensions,
      });
      navigate('/results');
    } catch (err) {
      setApiError(extractApiErrorMessage(err));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="page-container">
      <div className="page-header">
        <div>
          <h1>Dermoscopic Image Upload &amp; Validation</h1>
          <p>
            Upload a skin lesion image for binary classification (Benign vs. Malignant) and
            Grad-CAM visual explainability.
          </p>
        </div>
      </div>

      <DisclaimerBanner compact />

      {!health?.model_loaded && (
        <div className="alert-box alert-warning">
          <AlertCircle size={20} />
          <div>
            <strong>Trained Model Weights Not Found Yet:</strong> The backend is online, but{' '}
            <code>backend/models/efficientnetb0_ham10000_binary.keras</code> has not been generated
            yet. Train the model using <code>.venv/bin/python scripts/train_model.py</code> before
            running live predictions.
          </div>
        </div>
      )}

      {validationError && (
        <div className="alert-box alert-error" role="alert">
          <AlertCircle size={20} />
          <div>
            <strong>File Validation Error: </strong>
            <span>{validationError}</span>
          </div>
        </div>
      )}

      {apiError && (
        <div className="alert-box alert-error" role="alert">
          <AlertCircle size={20} />
          <div>
            <strong>Inference Pipeline Error: </strong>
            <span>{apiError}</span>
          </div>
        </div>
      )}

      <div className="upload-workspace-grid">
        {/* Left Column: Dropzone or Image Preview */}
        <section className="card">
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden-input"
            onChange={(e) => validateAndSelectFile(e.target.files?.[0])}
          />

          {!selectedFile ? (
            <div
              className={`upload-dropzone-large ${dragActive ? 'drag-active' : ''}`}
              onDragOver={(e) => {
                e.preventDefault();
                setDragActive(true);
              }}
              onDragLeave={() => setDragActive(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragActive(false);
                if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                  validateAndSelectFile(e.dataTransfer.files[0]);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
              }}
            >
              <div className="dropzone-icon-circle large">
                <UploadCloud size={38} />
              </div>
              <h2>Drag &amp; Drop Dermoscopic Image Here</h2>
              <p>or click below to browse your local filesystem</p>
              <button type="button" className="btn btn-primary">
                <FileImage size={17} />
                <span>Browse Image Files</span>
              </button>
              <div className="upload-constraints-note">
                <span>Supported formats: JPG, JPEG, PNG</span>
                <span>•</span>
                <span>Min resolution: 32×32 px</span>
                <span>•</span>
                <span>Max file size: 10 MB</span>
              </div>
            </div>
          ) : (
            <div className="preview-panel">
              <div className="preview-header">
                <div className="preview-status-badge">
                  <CheckCircle2 size={16} />
                  <span>Validated Dermoscopic Input Ready</span>
                </div>
                <div className="preview-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isSubmitting}
                  >
                    <RefreshCw size={14} />
                    <span>Replace Image</span>
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger-outline btn-sm"
                    onClick={handleClearFile}
                    disabled={isSubmitting}
                  >
                    <Trash2 size={14} />
                    <span>Remove</span>
                  </button>
                </div>
              </div>

              <div className="preview-image-frame">
                <img src={previewUrl} alt="Selected skin lesion preview" className="preview-img" />
              </div>

              <div className="preview-meta-grid">
                <div className="meta-item">
                  <span className="meta-label">File Name</span>
                  <span className="meta-val font-mono">{selectedFile.name}</span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">File Size</span>
                  <span className="meta-val font-mono">
                    {(selectedFile.size / 1024).toFixed(1)} KB
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Original Resolution</span>
                  <span className="meta-val font-mono">
                    {imageDimensions
                      ? `${imageDimensions.width} × ${imageDimensions.height} px`
                      : 'Detecting...'}
                  </span>
                </div>
                <div className="meta-item">
                  <span className="meta-label">Model Input Tensor</span>
                  <span className="meta-val font-mono">224 × 224 × 3 RGB</span>
                </div>
              </div>

              <div className="submit-inference-bar">
                <button
                  type="button"
                  className="btn btn-primary btn-lg btn-full"
                  onClick={handleRunInference}
                  disabled={isSubmitting}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 size={20} className="spin-icon" />
                      <span>Running EfficientNetB0 Inference &amp; Grad-CAM...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={19} />
                      <span>Analyze Lesion &amp; Generate Grad-CAM</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          )}
        </section>

        {/* Right Column: Preprocessing & Acquisition Guidelines */}
        <aside className="card">
          <div className="card-header">
            <FileImage size={20} className="card-header-icon" />
            <div>
              <h2>Preprocessing &amp; Input Protocol</h2>
              <p className="card-subtitle">Automated pipeline applied before inference</p>
            </div>
          </div>

          <ol className="protocol-steps">
            <li>
              <strong>Security &amp; Integrity Verification:</strong> Uploaded file bytes are
              verified with Pillow (<code>Image.verify()</code> and <code>Image.load()</code>).
              User-supplied filenames are sanitized.
            </li>
            <li>
              <strong>RGB Color Standardization:</strong> Single-channel grayscale or 4-channel
              RGBA images are converted strictly to 3-channel RGB.
            </li>
            <li>
              <strong>Spatial Resizing:</strong> Images are resized using bilinear interpolation to{' '}
              <code>224 × 224</code> pixels to match the <code>EfficientNetB0</code> input tensor.
            </li>
            <li>
              <strong>Identical Normalization:</strong> Pixel values are passed through{' '}
              <code>tf.keras.applications.efficientnet.preprocess_input</code> identically to the
              training pipeline. No random augmentations are applied during inference.
            </li>
            <li>
              <strong>Grad-CAM Activation Extraction:</strong> Gradients of the predicted class
              score with respect to <code>top_activation</code> feature maps are pooled and
              projected onto the image.
            </li>
          </ol>
        </aside>
      </div>
    </div>
  );
}
