import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Cpu,
  Eye,
  FileText,
  Layers,
  ShieldCheck,
  UploadCloud,
} from 'lucide-react';
import DisclaimerBanner from '../components/DisclaimerBanner';

export default function HomeDashboard({ modelInfo, health, onQuickSelectFile }) {
  const navigate = useNavigate();
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFileChosen = (file) => {
    if (!file) return;
    if (onQuickSelectFile) {
      onQuickSelectFile(file);
    }
    navigate('/upload');
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChosen(e.dataTransfer.files[0]);
    }
  };

  const workflowSteps = [
    {
      step: '01',
      title: 'Dermoscopic Image Input',
      desc: 'Upload a dermoscopic skin lesion image (JPG, JPEG, or PNG up to 10 MB). Validated for integrity and dimensions.',
      icon: UploadCloud,
    },
    {
      step: '02',
      title: 'Standardized Preprocessing',
      desc: 'Converted to 3-channel RGB, resized to 224 × 224 pixels, and normalized via EfficientNet preprocessing.',
      icon: Layers,
    },
    {
      step: '03',
      title: 'EfficientNetB0 Binary Inference',
      desc: 'Transfer-learned CNN estimates Benign vs. Malignant probabilities using a validation-calibrated decision threshold.',
      icon: BrainCircuit,
    },
    {
      step: '04',
      title: 'Grad-CAM Explainable AI',
      desc: 'Computes spatial activation heatmaps from the top_activation convolutional layer to highlight influential regions.',
      icon: Eye,
    },
    {
      step: '05',
      title: 'Downloadable PDF Report',
      desc: 'Generates a structured academic report with probabilities, Grad-CAM overlays, and clinical disclaimers.',
      icon: FileText,
    },
  ];

  const threshold = modelInfo?.decision_threshold ?? 0.5;
  const inputShape = modelInfo?.input_dimensions?.shape?.join(' × ') || '224 × 224 × 3';

  return (
    <div className="page-container">
      {/* Hero Banner */}
      <section className="hero-card">
        <div className="hero-content">
          <div className="academic-badge">
            <span>Graphic Era Hill University • B.Tech AI &amp; ML Major Project • Team CSE27-304</span>
          </div>
          <h1 className="hero-title">
            Skin Disease Classification Using Deep Learning on the HAM10000 Dataset
          </h1>
          <p className="hero-description">
            An end-to-end computer vision and explainable deep learning system that analyzes
            dermoscopic skin lesion images and classifies them into{' '}
            <strong className="text-benign">Benign (Non-cancerous)</strong> or{' '}
            <strong className="text-malignant">Malignant (Cancerous)</strong> categories using
            transfer learning with <code>EfficientNetB0</code> and <code>Grad-CAM</code> visual
            interpretability.
          </p>

          <div className="hero-actions">
            <button
              type="button"
              className="btn btn-primary btn-lg"
              onClick={() => navigate('/upload')}
            >
              <UploadCloud size={19} />
              <span>Upload Dermoscopic Image</span>
              <ArrowRight size={17} />
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-lg"
              onClick={() => navigate('/performance')}
            >
              <Activity size={18} />
              <span>View Test-Set Evaluation</span>
            </button>
          </div>
        </div>

        {/* Quick Drag-and-Drop Card on Home Dashboard */}
        <div
          className={`quick-dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
          }}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="hidden-input"
            onChange={(e) => handleFileChosen(e.target.files?.[0])}
          />
          <div className="dropzone-icon-circle">
            <UploadCloud size={30} />
          </div>
          <h3>Quick Dermoscopic Dropzone</h3>
          <p>Drag &amp; drop a skin lesion image here, or click to browse files</p>
          <div className="file-type-chips">
            <span className="chip">JPG / JPEG</span>
            <span className="chip">PNG</span>
            <span className="chip">Max 10 MB</span>
            <span className="chip">224×224 RGB Auto-Resize</span>
          </div>
        </div>
      </section>

      {/* Medical Disclaimer */}
      <DisclaimerBanner />

      {/* Model Information & Binary Mapping Overview Grid */}
      <div className="dashboard-grid-2col">
        <section className="card">
          <div className="card-header">
            <Cpu size={20} className="card-header-icon" />
            <div>
              <h2>Model Architecture &amp; Runtime Specification</h2>
              <p className="card-subtitle">Live configuration loaded from Flask inference backend</p>
            </div>
          </div>

          <div className="spec-list">
            <div className="spec-row">
              <span className="spec-label">Primary Architecture</span>
              <span className="spec-value font-mono">
                {modelInfo?.model_architecture || 'EfficientNetB0'} (Transfer Learning)
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Classification Head</span>
              <span className="spec-value font-mono">
                GlobalAvgPool2D → Dropout(0.35) → Dense(1, Sigmoid)
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Input Tensor Dimensions</span>
              <span className="spec-value font-mono">{inputShape} (RGB)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Target Output Classes</span>
              <span className="spec-value">
                <span className="badge badge-benign">0: Benign</span>{' '}
                <span className="badge badge-malignant">1: Malignant</span>
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Active Decision Threshold</span>
              <span className="spec-value font-mono">
                P(Malignant) &ge; {Number(threshold).toFixed(2)}
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Explainable AI Layer</span>
              <span className="spec-value font-mono">Grad-CAM on top_activation (7×7×1280)</span>
            </div>
            <div className="spec-row">
              <span className="spec-label">Inference Engine Status</span>
              <span className="spec-value">
                {health?.model_loaded ? (
                  <span className="status-text-ready">
                    <CheckCircle2 size={15} /> Loaded &amp; Ready
                  </span>
                ) : (
                  <span className="status-text-warn">
                    Awaiting trained weights (run scripts/train_model.py)
                  </span>
                )}
              </span>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="card-header">
            <ShieldCheck size={20} className="card-header-icon" />
            <div>
              <h2>HAM10000 Binary Classification Mapping</h2>
              <p className="card-subtitle">
                Explicit clinical grouping of the 7 original diagnostic labels
              </p>
            </div>
          </div>

          <div className="mapping-summary-boxes">
            <div className="mapping-box benign-box">
              <div className="mapping-box-header">
                <span className="badge badge-benign">Class 0 — Benign (Non-Cancerous)</span>
                <span className="mapping-count">4 Original Categories</span>
              </div>
              <ul className="mapping-list">
                <li>
                  <code>nv</code> — Melanocytic nevi (benign moles)
                </li>
                <li>
                  <code>bkl</code> — Benign keratosis-like lesions (seborrheic keratosis, solar
                  lentigo)
                </li>
                <li>
                  <code>df</code> — Dermatofibroma (benign dermal nodule)
                </li>
                <li>
                  <code>vasc</code> — Vascular lesions (angiomas, angiokeratomas, hemorrhage)
                </li>
              </ul>
            </div>

            <div className="mapping-box malignant-box">
              <div className="mapping-box-header">
                <span className="badge badge-malignant">Class 1 — Malignant (Cancerous)</span>
                <span className="mapping-count">3 Original Categories</span>
              </div>
              <ul className="mapping-list">
                <li>
                  <code>mel</code> — Malignant Melanoma
                </li>
                <li>
                  <code>bcc</code> — Basal Cell Carcinoma
                </li>
                <li>
                  <code>akiec</code> — Actinic keratoses &amp; intraepithelial carcinoma (Bowen&apos;s
                  disease / SCC in situ)
                </li>
              </ul>
            </div>
          </div>
        </section>
      </div>

      {/* End-to-End Project Workflow Overview */}
      <section className="card">
        <div className="card-header">
          <BrainCircuit size={20} className="card-header-icon" />
          <div>
            <h2>End-to-End Diagnostic Research Workflow</h2>
            <p className="card-subtitle">
              Leakage-free pipeline from dermoscopic acquisition to explainable AI reporting
            </p>
          </div>
        </div>

        <div className="workflow-grid">
          {workflowSteps.map((item) => {
            const StepIcon = item.icon;
            return (
              <div key={item.step} className="workflow-step-card">
                <div className="workflow-step-top">
                  <span className="workflow-step-num">{item.step}</span>
                  <div className="workflow-step-icon">
                    <StepIcon size={20} />
                  </div>
                </div>
                <h3>{item.title}</h3>
                <p>{item.desc}</p>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
