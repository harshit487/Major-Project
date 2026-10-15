import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  Download,
  Eye,
  FileText,
  Info,
  Loader2,
  RotateCcw,
  ShieldAlert,
  UploadCloud,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import DisclaimerBanner from '../components/DisclaimerBanner';
import {
  downloadPredictionReport,
  extractApiErrorMessage,
  resolveBackendAssetUrl,
} from '../services/api';

export default function PredictionResultsPage({ predictionResult, onResetPrediction }) {
  const navigate = useNavigate();
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState(null);

  if (!predictionResult) {
    return (
      <div className="page-container">
        <div className="empty-state-card card">
          <div className="dropzone-icon-circle large">
            <UploadCloud size={36} />
          </div>
          <h2>No Active Prediction Result</h2>
          <p>
            No dermoscopic image has been analyzed in this session yet. Upload an image to run real
            EfficientNetB0 binary inference and Grad-CAM visualization.
          </p>
          <button
            type="button"
            className="btn btn-primary btn-lg"
            onClick={() => navigate('/upload')}
          >
            <UploadCloud size={18} />
            <span>Go to Image Upload</span>
          </button>
        </div>
      </div>
    );
  }

  const isMalignant =
    String(predictionResult.predicted_class).toLowerCase() === 'malignant';
  const benignProbPct = Number((predictionResult.benign_probability * 100).toFixed(2));
  const malignantProbPct = Number((predictionResult.malignant_probability * 100).toFixed(2));
  const confidencePct = Number((predictionResult.confidence_score * 100).toFixed(2));
  const thresholdPct = Number(((predictionResult.decision_threshold ?? 0.5) * 100).toFixed(1));

  const chartData = [
    {
      name: 'Benign (Non-Cancerous)',
      probability: benignProbPct,
      fill: '#0d9488',
    },
    {
      name: 'Malignant (Cancerous)',
      probability: malignantProbPct,
      fill: '#dc2626',
    },
  ];

  const vis = predictionResult.visualizations || {};
  const originalUrl =
    resolveBackendAssetUrl(vis.original_image_url) || predictionResult.clientPreviewUrl;
  const heatmapUrl = resolveBackendAssetUrl(vis.gradcam_heatmap_url);
  const overlayUrl = resolveBackendAssetUrl(vis.gradcam_overlay_url);

  const handleDownloadPdf = async () => {
    setDownloadingPdf(true);
    setPdfError(null);
    try {
      await downloadPredictionReport(predictionResult);
    } catch (err) {
      setPdfError(extractApiErrorMessage(err));
    } finally {
      setDownloadingPdf(false);
    }
  };

  const handleAnalyzeAnother = () => {
    if (onResetPrediction) onResetPrediction();
    navigate('/upload');
  };

  return (
    <div className="page-container">
      <div className="page-header results-header-row">
        <div>
          <h1>Dermoscopic Prediction &amp; Explainable AI Results</h1>
          <p className="font-mono text-sm">
            Prediction ID: {predictionResult.prediction_id} • Timestamp:{' '}
            {new Date(predictionResult.timestamp).toLocaleString()}
          </p>
        </div>
        <div className="header-button-group">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleAnalyzeAnother}
          >
            <RotateCcw size={16} />
            <span>Analyze Another Image</span>
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={handleDownloadPdf}
            disabled={downloadingPdf}
          >
            {downloadingPdf ? (
              <>
                <Loader2 size={16} className="spin-icon" />
                <span>Generating PDF...</span>
              </>
            ) : (
              <>
                <Download size={16} />
                <span>Download PDF Report</span>
              </>
            )}
          </button>
        </div>
      </div>

      {pdfError && (
        <div className="alert-box alert-error" role="alert">
          <AlertTriangle size={18} />
          <span>{pdfError}</span>
        </div>
      )}

      {/* Primary Verdict Banner */}
      <section
        className={`verdict-banner ${isMalignant ? 'verdict-malignant' : 'verdict-benign'}`}
      >
        <div className="verdict-left">
          <div className="verdict-icon-wrap">
            {isMalignant ? <ShieldAlert size={34} /> : <CheckCircle2 size={34} />}
          </div>
          <div>
            <span className="verdict-eyebrow">BINARY CLASSIFICATION OUTPUT</span>
            <h2 className="verdict-class-title">
              {predictionResult.predicted_class}{' '}
              <span className="verdict-sublabel">
                ({isMalignant ? 'Cancerous / Neoplastic Risk' : 'Non-Cancerous'})
              </span>
            </h2>
            <p className="verdict-calibration-note">
              <Info size={14} />
              <span>
                {predictionResult.calibration_note ||
                  'Uncalibrated model probability estimate. Not a clinically calibrated post-test probability.'}
              </span>
            </p>
          </div>
        </div>

        <div className="verdict-metrics-strip">
          <div className="verdict-stat-box">
            <span className="stat-label">Benign Probability</span>
            <span className="stat-value text-benign">{benignProbPct.toFixed(2)}%</span>
          </div>
          <div className="verdict-stat-box">
            <span className="stat-label">Malignant Probability</span>
            <span className="stat-value text-malignant">{malignantProbPct.toFixed(2)}%</span>
          </div>
          <div className="verdict-stat-box">
            <span className="stat-label">Estimated Confidence</span>
            <span className="stat-value">{confidencePct.toFixed(2)}%</span>
          </div>
          <div className="verdict-stat-box">
            <span className="stat-label">Decision Threshold</span>
            <span className="stat-value font-mono">
              {(predictionResult.decision_threshold ?? 0.5).toFixed(2)} ({thresholdPct}%)
            </span>
          </div>
        </div>
      </section>

      <DisclaimerBanner />

      {/* Probability Chart & Model Execution Metadata */}
      <div className="dashboard-grid-2col">
        <section className="card">
          <div className="card-header">
            <FileText size={20} className="card-header-icon" />
            <div>
              <h2>Binary Class Probability Distribution</h2>
              <p className="card-subtitle">
                P(Benign) + P(Malignant) = {(benignProbPct + malignantProbPct).toFixed(2)}% •
                Threshold dashed line at {thresholdPct}%
              </p>
            </div>
          </div>

          <div className="chart-wrapper-sm">
            <ResponsiveContainer width="100%" height={235}>
              <BarChart
                data={chartData}
                layout="vertical"
                margin={{ top: 12, right: 30, left: 20, bottom: 12 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  type="number"
                  domain={[0, 100]}
                  unit="%"
                  tick={{ fontSize: 12, fill: '#475569' }}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  width={155}
                  tick={{ fontSize: 12, fontWeight: 600, fill: '#0f172a' }}
                />
                <Tooltip formatter={(value) => [`${value}%`, 'Estimated Probability']} />
                <ReferenceLine
                  x={thresholdPct}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  label={{
                    value: `Threshold (${thresholdPct}%)`,
                    position: 'top',
                    fill: '#b45309',
                    fontSize: 11,
                  }}
                />
                <Bar dataKey="probability" radius={[0, 6, 6, 0]} barSize={34}>
                  {chartData.map((entry, idx) => (
                    <Cell key={`cell-${idx}`} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <Clock size={20} className="card-header-icon" />
            <div>
              <h2>Inference Audit Details</h2>
              <p className="card-subtitle">Traceable metadata for this prediction session</p>
            </div>
          </div>

          <div className="spec-list">
            <div className="spec-row">
              <span className="spec-label">Prediction Identifier</span>
              <span className="spec-value font-mono">{predictionResult.prediction_id}</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Model Backbone</span>
              <span className="spec-value font-mono">
                {predictionResult.model_architecture || 'EfficientNetB0'}
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Preprocessed Input Tensor</span>
              <span className="spec-value font-mono">
                {(predictionResult.input_dimensions || [224, 224, 3]).join(' × ')} RGB
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Grad-CAM Target Layer</span>
              <span className="spec-value font-mono">{vis.target_layer || 'top_activation'}</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Decision Rule</span>
              <span className="spec-value font-mono">
                Malignant if P(Malignant) &ge;{' '}
                {(predictionResult.decision_threshold ?? 0.5).toFixed(2)}
              </span>
            </div>
          </div>
        </section>
      </div>

      {/* 3-Panel Grad-CAM Visual Comparison */}
      <section className="card">
        <div className="card-header">
          <Eye size={20} className="card-header-icon" />
          <div>
            <h2>Explainable AI (Grad-CAM) Visual Comparison Panel</h2>
            <p className="card-subtitle">
              Side-by-side inspection of the original resized input, convolutional activation
              heatmap, and blended overlay
            </p>
          </div>
        </div>

        <div className="gradcam-comparison-grid">
          <div className="gradcam-panel">
            <div className="gradcam-panel-header">
              <span>1. Original Uploaded Image (224×224)</span>
            </div>
            <div className="gradcam-img-box">
              {originalUrl ? (
                <img src={originalUrl} alt="Original dermoscopic lesion" />
              ) : (
                <span className="text-muted">Original image unavailable</span>
              )}
            </div>
            <p className="gradcam-caption">
              Standardized 224 × 224 RGB dermoscopic input passed to EfficientNetB0.
            </p>
          </div>

          <div className="gradcam-panel">
            <div className="gradcam-panel-header">
              <span>2. Grad-CAM Activation Heatmap</span>
            </div>
            <div className="gradcam-img-box">
              {heatmapUrl ? (
                <img src={heatmapUrl} alt="Grad-CAM activation heatmap" />
              ) : (
                <span className="text-muted">Heatmap unavailable</span>
              )}
            </div>
            <p className="gradcam-caption">
              Spatial importance map from <code>{vis.target_layer || 'top_activation'}</code> (Red =
              high activation, Blue = low).
            </p>
          </div>

          <div className="gradcam-panel">
            <div className="gradcam-panel-header">
              <span>3. Blended Grad-CAM Overlay</span>
            </div>
            <div className="gradcam-img-box">
              {overlayUrl ? (
                <img src={overlayUrl} alt="Grad-CAM overlay on original lesion" />
              ) : (
                <span className="text-muted">Overlay unavailable</span>
              )}
            </div>
            <p className="gradcam-caption">
              Alpha-blended overlay localizing the lesion regions driving the model&apos;s output.
            </p>
          </div>
        </div>

        <div className="gradcam-explanation-box">
          <strong>How to Interpret Grad-CAM Carefully: </strong>
          <span>
            {vis.explanation ||
              'Grad-CAM identifies spatial image regions associated with the deep learning model’s prediction; it does not establish a medical diagnosis or prove that the highlighted region is clinically correct.'}
          </span>
        </div>
      </section>
    </div>
  );
}
